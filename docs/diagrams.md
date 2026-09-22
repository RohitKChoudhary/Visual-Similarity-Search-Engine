# Design Diagrams

These render automatically when viewing this file on GitHub. For the
PDF report, paste any of the code blocks below into https://mermaid.live
and export as PNG/SVG.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Input
        IMG[Image Dataset<br/>data/images/&lt;class&gt;/*.jpg]
        QIMG[Query Image]
        QTXT[Query Text]
    end

    subgraph Core["Visual Similarity Search Engine"]
        EMB[Embedder Module<br/>ClipEmbedder<br/>CLIP ViT-B/32]
        IDX[Indexer Module<br/>FaissIndexer<br/>IndexFlatIP]
        SRCH[Search Module<br/>search]
        EVAL[Evaluation Module<br/>precision_at_k]
    end

    subgraph Storage
        FIDX[(faiss.index)]
        META[(metadata.json)]
    end

    OUT[Ranked Results<br/>image path + similarity score]

    IMG --> EMB
    QIMG --> EMB
    QTXT --> EMB
    EMB -->|image/text embeddings| IDX
    IDX --> FIDX
    IDX --> META
    EMB -->|query embedding| SRCH
    FIDX --> SRCH
    META --> SRCH
    SRCH --> OUT
    FIDX --> EVAL
    META --> EVAL
    EVAL --> OUT
```

---

## 2. Workflow / Process Flow

```mermaid
flowchart LR
    A[Start] --> B{Index exists?}
    B -- No --> C["python cli.py build-index"]
    C --> D[Load CLIP model]
    D --> E[Load & embed all images]
    E --> F[Build FAISS index]
    F --> G[Save index + metadata]
    G --> H{Index exists}
    B -- Yes --> H
    H --> I{Command}
    I -- search --> J["Embed query image"]
    I -- search-text --> K["Embed query text"]
    I -- evaluate --> L["Re-embed dataset, compute Precision@K"]
    J --> M[FAISS similarity search]
    K --> M
    M --> N[Print ranked results]
    L --> O[Print Precision@K score]
    N --> P[End]
    O --> P
```

---

## 3. Use Case Diagram

```mermaid
flowchart LR
    User((User))

    subgraph System["Visual Similarity Search Engine"]
        UC1([Build Index])
        UC2([Search by Image])
        UC3([Search by Text])
        UC4([Evaluate Retrieval Quality])
    end

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    UC2 -.->|includes| UC5([Embed Query])
    UC3 -.->|includes| UC5
    UC4 -.->|includes| UC6([Re-embed Dataset])
```

---

## 4. Sequence Diagram — Image Search

```mermaid
sequenceDiagram
    actor User
    participant CLI as cli.py
    participant Embedder as ClipEmbedder
    participant Indexer as FaissIndexer
    participant FAISS as FAISS Index

    User->>CLI: python cli.py search --query img.jpg --top-k 5
    CLI->>Embedder: embed_images([query_image])
    Embedder->>Embedder: CLIP forward pass
    Embedder-->>CLI: query_embedding
    CLI->>Indexer: FaissIndexer.load(index_path, metadata_path)
    Indexer-->>CLI: loaded index + image_paths
    CLI->>FAISS: search(query_embedding, top_k)
    FAISS-->>CLI: (scores, indices)
    CLI->>CLI: map indices -> image_paths
    CLI-->>User: ranked list of (path, similarity)
```

---

## 5. Class / Component Diagram

```mermaid
classDiagram
    class ClipEmbedder {
        -model: CLIPModel
        -processor: CLIPProcessor
        -device: str
        +embed_images(images, batch_size) ndarray
        +embed_text(text) ndarray
    }

    class FaissIndexer {
        -index: faiss.IndexFlatIP
        -image_paths: List~str~
        +build(embeddings, image_paths)
        +save(index_path, metadata_path)
        +load(index_path, metadata_path)$ FaissIndexer
    }

    class SearchModule {
        <<module: search.py>>
        +search(indexer, query_embedding, top_k) List
    }

    class EvaluateModule {
        <<module: evaluate.py>>
        +precision_at_k(indexer, embeddings, k) tuple
    }

    class CLI {
        <<module: cli.py>>
        +cmd_build_index(args)
        +cmd_search(args)
        +cmd_search_text(args)
        +cmd_evaluate(args)
    }

    CLI --> ClipEmbedder : uses
    CLI --> FaissIndexer : uses
    CLI --> SearchModule : uses
    CLI --> EvaluateModule : uses
    SearchModule --> FaissIndexer : queries
    EvaluateModule --> FaissIndexer : queries
```

---

## 6. Data / Storage Schema

No relational database is used — persistence is a FAISS binary index
paired with a JSON metadata sidecar. Schema shown as an ER-style
diagram for documentation purposes:

```mermaid
erDiagram
    IMAGE_RECORD {
        int vector_id PK "position in FAISS index"
        string image_path "relative path, stored in metadata.json"
        float_array embedding "512-dim CLIP embedding, stored in faiss.index"
    }
```

**`index/faiss.index`** — binary FAISS `IndexFlatIP` file; row `i` holds the
L2-normalized 512-dim embedding for the `i`-th image.

**`index/metadata.json`**:
```json
{
  "image_paths": ["data/images/rose/10503217854_e66a804309.jpg", "..."]
}
```
`image_paths[i]` corresponds to embedding row `i` in `faiss.index` — this
positional mapping is the join key between the two files.
