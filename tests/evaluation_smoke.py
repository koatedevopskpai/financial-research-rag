"""Execute the actual evaluation CLI and verify each exit status/report pairing."""

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/S5-ci")
    output.mkdir(parents=True, exist_ok=True)
    for fault in ("none", "retrieval", "citation", "abstention"):
        report = output / f"{fault}.json"
        command = [
            sys.executable,
            "-m",
            "app.evaluation",
            "--fault",
            fault,
            "--output",
            str(report),
        ]
        result = subprocess.run(command, check=False)
        assert result.returncode == (0 if fault == "none" else 1)
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["gate"]["passed"] == (fault == "none")
        assert payload["fault_injection"] == fault
        assert payload["sample_size"] == 10
        assert payload["execution_mode"] == "offline_local_bm25_extractive"
        if fault == "retrieval":
            assert "recall_at_5" in payload["gate"]["failures"]
        if fault in {"citation", "abstention"}:
            assert "answer_contract_passed" in payload["gate"]["failures"]
            assert payload["summary"]["retrieval"]["recall_at_5"] == 1.0
    print("S5 CLI: baseline accepted; retrieval, citation and abstention faults rejected.")


if __name__ == "__main__":
    main()
