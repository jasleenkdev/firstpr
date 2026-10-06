"""FirstPR serving. Runtime modules (encoder, catalog, ranking, github, explain, app) import only
numpy, requests and fastapi so the API stays small; `build` is build-time only (torch)."""
