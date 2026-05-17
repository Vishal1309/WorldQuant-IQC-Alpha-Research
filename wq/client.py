"""
WorldQuant Brain API client.

Authenticates via HTTP Basic Auth against https://api.worldquantbrain.com,
then reuses the resulting session cookie for all subsequent calls. This is the
standard pattern used by all Brain API clients (including pyworldquant itself).

Usage:
    client = get_client()             # authenticates, returns requests.Session
    fields_df = fetch_datafields(client)
    result    = simulate(client, config)
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path

import requests
from requests.exceptions import HTTPError, Timeout, ConnectionError as ReqConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

load_dotenv()

API_BASE = "https://api.worldquantbrain.com"
DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Session with base URL support
# ---------------------------------------------------------------------------

class BrainSession(requests.Session):
    """requests.Session that prepends API_BASE to every relative URL."""

    def request(self, method, url, *args, **kwargs):
        if not url.startswith("http"):
            url = API_BASE + url
        return super().request(method, url, *args, **kwargs)


# ---------------------------------------------------------------------------
# Auth + session
# ---------------------------------------------------------------------------

def get_client() -> BrainSession:
    """
    Authenticate with WorldQuant Brain and return an authenticated Session.

    Reads WQ_USERNAME and WQ_PASSWORD from .env (or environment variables).
    Brain uses HTTP Basic Auth to establish a session cookie — the Session
    object keeps that cookie alive for all subsequent requests automatically.
    """
    username = os.environ.get("WQ_USERNAME")
    password = os.environ.get("WQ_PASSWORD")
    if not username or not password:
        raise EnvironmentError(
            "WQ_USERNAME and WQ_PASSWORD must be set in your .env file. "
            "Copy .env.example to .env and fill in your Brain credentials."
        )

    session = BrainSession()
    resp = session.post("/authentication", auth=(username, password))

    if resp.status_code == 401:
        raise PermissionError("Brain authentication failed — check WQ_USERNAME and WQ_PASSWORD.")
    resp.raise_for_status()

    print(f"Authenticated as {username}")
    return session


# ---------------------------------------------------------------------------
# Retry decorator for transient errors (429, 503, timeouts)
# ---------------------------------------------------------------------------

_retry = retry(
    retry=retry_if_exception_type((HTTPError, Timeout, ReqConnectionError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Data field catalog
# ---------------------------------------------------------------------------

@_retry
def _get_page(
    session: requests.Session,
    offset: int,
    limit: int = 20,
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
) -> dict:
    resp = session.get(
        "/data-fields",
        params={
            "instrumentType": "EQUITY",
            "region": region,
            "delay": delay,
            "universe": universe,
            "offset": offset,
            "limit": limit,
        },
        timeout=30,
    )
    if not resp.ok:
        raise requests.exceptions.HTTPError(
            f"{resp.status_code} — {resp.text[:300]}", response=resp
        )
    return resp.json()


def fetch_datafields(
    session: requests.Session,
    save: bool = True,
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
) -> pd.DataFrame:
    """
    Download the complete data field catalog from Brain (120k+ fields).

    Paginates through all pages and caches results to data/datafields.parquet.
    First run takes 5–15 minutes; subsequent runs load from cache instantly.

    Returns:
        DataFrame with columns: id, name, description, type, category,
        subcategory, region, delay, alphaCount
    """
    DATA_DIR.mkdir(exist_ok=True)

    kwargs = dict(region=region, delay=delay, universe=universe)
    first_page = _get_page(session, offset=0, limit=1, **kwargs)
    total = first_page.get("count", 0)
    print(f"Fetching {total:,} data fields from Brain...")

    records = []
    page_size = 5
    with tqdm(total=total, unit="fields") as pbar:
        offset = 0
        while offset < total:
            page = _get_page(session, offset=offset, limit=page_size, **kwargs)
            batch = page.get("results", [])
            if not batch:
                break
            records.extend(batch)
            pbar.update(len(batch))
            offset += len(batch)

    df = pd.json_normalize(records)
    df.columns = [c.lower().replace(".", "_") for c in df.columns]

    if save:
        path = DATA_DIR / "datafields.parquet"
        df.to_parquet(path, index=False)
        print(f"Saved {len(df):,} fields → {path}")

    return df


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

@_retry
def fetch_operators(session: requests.Session, save: bool = True) -> list[dict]:
    """
    Download the list of all Brain alpha operators.

    Returns:
        List of operator dicts with keys: name, description, category, syntax
    """
    resp = session.get("/operators", timeout=30)
    resp.raise_for_status()
    operators = resp.json()

    if save:
        DATA_DIR.mkdir(exist_ok=True)
        path = DATA_DIR / "operators.json"
        path.write_text(json.dumps(operators, indent=2))
        print(f"Saved {len(operators)} operators → {path}")

    return operators


# ---------------------------------------------------------------------------
# Alpha simulation
# ---------------------------------------------------------------------------

@_retry
def fetch_alpha(session: requests.Session, alpha_id: str) -> dict:
    """
    Fetch a completed alpha record from /alphas/{id}.

    Returns the full alpha dict including is/os performance metrics.
    Raises HTTPError (with response body) on failure so auth issues are visible.
    """
    resp = session.get(f"/alphas/{alpha_id}", timeout=30)
    if not resp.ok:
        raise requests.exceptions.HTTPError(
            f"{resp.status_code} fetching alpha {alpha_id} — {resp.text[:300]}",
            response=resp,
        )
    return resp.json()


@_retry
def submit_alpha(session: requests.Session, config) -> str:
    """
    Submit an alpha to Brain for simulation. Returns the simulation ID.

    Brain returns HTTP 202 with a Location header pointing to the simulation URL.
    """
    resp = session.post("/simulations", json=config.to_payload(), timeout=30)
    if not resp.ok:
        raise requests.exceptions.HTTPError(
            f"{resp.status_code} — {resp.text[:500]}", response=resp
        )

    # Simulation URL is in the Location header: /simulations/<id>
    location = resp.headers.get("Location", "")
    sim_id = location.split("/")[-1]
    if not sim_id:
        sim_id = resp.json().get("id", "")
    return sim_id


def poll_result(session: requests.Session, sim_id: str, poll_interval: int = 10, timeout: int = 600) -> dict:
    """
    Poll a simulation until it reaches a terminal state (COMPLETE/ERROR/FAILED).

    When COMPLETE, automatically fetches the alpha record from /alphas/{id} and
    merges the is/os metric blocks into the returned dict so callers always get
    a uniform shape with sharpe, fitness, returns, turnover, drawdown.

    Args:
        session:       authenticated session from get_client()
        sim_id:        simulation ID returned by submit_alpha()
        poll_interval: seconds between polls (default 10)
        timeout:       max seconds to wait (default 600)

    Returns:
        Simulation dict enriched with is/os metrics from the alpha record.
    """
    terminal = {"COMPLETE", "ERROR", "FAILED", "WARNING"}
    elapsed = 0

    while elapsed < timeout:
        resp = session.get(f"/simulations/{sim_id}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "").upper()

        if status in terminal:
            # Brain returns the alpha ID as a string once COMPLETE — fetch its metrics
            alpha = data.get("alpha")
            if isinstance(alpha, str) and alpha:
                alpha_data = fetch_alpha(session, alpha)
                for key in ("is", "os"):
                    if key in alpha_data:
                        data[key] = alpha_data[key]
                data["alpha"] = alpha_data
            return data

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Simulation {sim_id} did not complete within {timeout}s.")


def simulate(session: requests.Session, config, save: bool = True) -> dict:
    """
    Submit an alpha and wait for the result in one call.

    Args:
        session: authenticated session from get_client()
        config:  AlphaConfig instance
        save:    append result to data/results.csv (default True)

    Returns:
        Raw Brain result dict. Use wq.utils.parse_result() to extract metrics.
    """
    print(f"Submitting: {config.expression[:80]}")
    sim_id = submit_alpha(session, config)
    print(f"  sim_id={sim_id} — polling every 10s...")

    result = poll_result(session, sim_id)
    status = result.get("status", "").upper()
    print(f"  status={status}")

    if save:
        from wq.utils import save_result
        save_result(result, config)

    return result
