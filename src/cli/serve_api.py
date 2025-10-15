from __future__ import annotations
import uvicorn

def main():
    import argparse
    p = argparse.ArgumentParser(description="Serve FastAPI for inference.")
    p.add_argument("--config", default="configs/default.yaml")  # kept for symmetry; not used directly
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run("src.chronic_risk.api:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
