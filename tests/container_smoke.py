"""Unconfigured local container smoke: live endpoints must fail closed, not pretend ready."""

import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> None:
    output = Path(sys.argv[1])
    for _ in range(30):
        try:
            with urlopen("http://127.0.0.1:8007/health", timeout=3) as response:
                health = json.load(response)
            break
        except (URLError, TimeoutError, ConnectionResetError):
            time.sleep(1)
    else:
        raise RuntimeError("container did not start")
    assert health["status"] == "degraded"
    assert health["document_pipeline_ready"] is False
    assert health["query_pipeline_ready"] is False
    req = Request(
        "http://127.0.0.1:8007/query",
        data=b'{"query":"What is revenue?","stream":false}',
        method="POST",
        headers={"content-type": "application/json"},
    )
    try:
        urlopen(req, timeout=5)
    except HTTPError as exc:
        assert exc.code == 503
    else:
        raise AssertionError("unconfigured Azure query path was accepted")
    output.write_text(
        json.dumps(
            {
                "execution_mode": "unconfigured_azure_container_smoke",
                "health": health,
                "query_http_status": 503,
                "external_provider_calls": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "S1 container starts without Azure config; degraded health; "
        "query fails closed with HTTP 503."
    )


if __name__ == "__main__":
    main()
