# Project Report Outline — fill this in yourself, then export to PDF

Don't just dump this structure with filler text — the plagiarism/AI-detection
pipeline is real. Use this as a skeleton, write the actual content in your
own words, based on what YOU built and what YOU observed running it.

1. **Cover Page** — title, your name, reg no, course, date

2. **Introduction** — 1 short paragraph: what the project is, why CV/retrieval
   is a relevant problem, one-line summary of the approach (CLIP + FAISS)

3. **Problem Statement** — copy/adapt from statement.md, expand slightly

4. **Functional Requirements** — list the 4 CLI commands as the 3+ functional
   modules: embedding extraction, indexing, search (image + text), evaluation

5. **Non-functional Requirements** — pull the table from README.md, but
   write 1-2 sentences justifying EACH one specific to your actual dataset
   size / run times once you've tested it

6. **System Architecture** — draw a simple box diagram:
   `Images -> CLIP Embedder -> FAISS Index <-> Search Module -> Results`
   with the text query path branching into the embedder too.
   (Use draw.io / excalidraw / even a simple diagram in your CV tool of choice —
   don't AI-generate this, sketch it based on your actual src/ layout)

7. **Design Diagrams** — all done, in `docs/diagrams.md` (renders on GitHub).
   For the PDF: paste each Mermaid block into https://mermaid.live, export
   as PNG, drop into the report. Covers: system architecture, workflow,
   use case, sequence (image search flow), class/component, and the
   FAISS index + metadata.json storage schema.

8. **Design Decisions & Rationale** — explain WHY pretrained CLIP over
   training from scratch (time, data size, overfitting risk), WHY FAISS
   IndexFlatIP for this scale and what you'd change at larger scale,
   WHY Precision@K as the metric

9. **Implementation Details** — briefly walk through each module
   (embedder, indexer, search, evaluate) — 1 paragraph each, referencing
   actual function names

10. **Screenshots / Results** — terminal output of build-index, search,
    search-text, and evaluate runs on YOUR actual dataset. Include the
    real Precision@K number you got.

11. **Testing Approach** — mention tests/test_pipeline.py (offline,
    synthetic-embedding tests for pipeline correctness) AND the
    Precision@K evaluation (end-to-end retrieval quality on real data)

12. **Challenges Faced** — write these honestly once you've actually run it
    (e.g. CLIP download size/time, choosing a good dataset, corrupt images,
    tuning top-k, etc.)

13. **Learnings & Key Takeaways** — your own words, based on what you
    actually learned building this

14. **Future Enhancements** — pull from README.md, expand with your own ideas

15. **References** — CLIP paper, FAISS paper/docs, HuggingFace transformers docs,
    dataset source
