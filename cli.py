\
import argparse
import json
import os
from typing import List, Dict

from resume_utils import parse_resume, rank_resume_against_many, match_resume_to_jd

def load_jd_items_from_file(path: str) -> List[Dict]:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Support {"items": [...]}
            if "items" in data and isinstance(data["items"], list):
                return data["items"]
            # Or {"job_descriptions": [...]}
            if "job_descriptions" in data and isinstance(data["job_descriptions"], list):
                return data["job_descriptions"]
        raise ValueError("JSON format not recognized. Expected a list or {'items': [...]}")
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        return [{"title": "JD", "description": txt}]
    else:
        raise ValueError("Unsupported JD file. Use .json or .txt")

def main():
    ap = argparse.ArgumentParser(description="Resume ↔ JD matcher (TF-IDF cosine similarity)")
    ap.add_argument("--resume", required=True, help="Path to resume (.pdf or .txt)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--jd-text", help="Job description as direct text")
    g.add_argument("--jd-file", help="Path to JD file (.json or .txt)")
    ap.add_argument("--top", type=int, default=5, help="Top K matches when using --jd-file")
    args = ap.parse_args()

    resume_text = parse_resume(args.resume)

    if args.jd_text:
        score = match_resume_to_jd(resume_text, args.jd_text)
        print(f"Match Score: {score:.2f}%")
        return

    items = load_jd_items_from_file(args.jd_file)
    results = rank_resume_against_many(resume_text, items, top_k=args.top)
    print(f"Top {len(results)} matches:")
    for idx, item, score in results:
        title = item.get("title") or f"JD #{idx+1}"
        print(f"- {title:50s}  {score:6.2f}%")

if __name__ == "__main__":
    main()
