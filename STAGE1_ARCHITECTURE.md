# STAGE 1: KNOWLEDGE GRAPH INDEX CONSTRUCTION

## 📋 TABLE OF CONTENTS
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Workflow](#workflow)
4. [Components](#components)
5. [Configuration](#configuration)
6. [Input/Output](#inputoutput)
7. [Metrics & Evaluation](#metrics--evaluation)
8. [Visualizations](#visualizations)
9. [Usage](#usage)
10. [Performance](#performance)

---

## OVERVIEW

### Purpose
Stage 1 builds the foundation of the Knowledge Graph from raw documents. It extracts structured knowledge (entities and relationships) and optionally generates Q&A pairs for RAG.

### Key Responsibilities
- **Knowledge Graph Construction**: Extract entities and relationships from text
- **Entity Recognition**: Identify named entities (NER)
- **Relationship Extraction**: Extract semantic relationships using OpenIE
- **Q&A Generation** (Optional): Create question-answer pairs for retrieval

### Pipeline Position
```
RAW DOCUMENTS
     ↓
[STAGE 1: KG Index] ← YOU ARE HERE
     ↓
tmp/kg_construction/*/hotpotqa/kg.txt
     ↓
[STAGE 2: Entity Resolution]
```

---

## ARCHITECTURE

### High-Level Design
```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: KG INDEX                        │
│                                                             │
│  Input: Dataset (documents)                                 │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   KGIndexer (Orchestrator)                           │  │
│  │                                                      │  │
│  │   ┌──────────────────┐   ┌──────────────────────┐   │  │
│  │   │ KGConstructor    │   │ QAConstructor        │   │  │
│  │   │ (Knowledge Graph)│   │ (Q&A Pairs)          │   │  │
│  │   └──────────────────┘   └──────────────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Output: kg.txt + Q&A pairs                                 │
└─────────────────────────────────────────────────────────────┘
```

### Component Hierarchy
```
KGIndexer (Main Orchestrator)
    ├─ KGConstructor
    │   ├─ NER Model (llm_ner_model)
    │   └─ OpenIE Model (llm_openie_model)
    └─ QAConstructor (Optional)
```

---

## WORKFLOW

### Main Execution Flow
```python
main(cfg) [entry point]
    │
    ├─► Load environment variables (.env)
    ├─► Initialize KGConstructor
    │    └─► Configure NER & OpenIE models
    ├─► Initialize QAConstructor (optional)
    └─► Create KGIndexer
         │
         └─► index_data(dataset)
              │
              ├─► Load dataset from cfg.dataset
              ├─► For each document:
              │    ├─► Extract entities (NER)
              │    ├─► Extract relationships (OpenIE)
              │    └─► Generate Q&A pairs (optional)
              └─► Save Knowledge Graph
```

### Detailed Steps

#### STEP 1: Configuration Loading
**File**: `gfmrag/workflow/config/stage1_index_dataset.yaml`

```yaml
defaults:
  - _self_
  - ner_model: llm_ner_model
  - openie_model: llm_openie_model

dataset:
  root: ./data
  data_name: hotpotqa

kg_constructor:
  _target_: gfmrag.kg_construction.KGConstructor
  open_ie_model: ${openie_model}
  ner_model: ${ner_model}
  num_processes: 10
  force: True
```

**Parameters**:
- `ner_model`: Named Entity Recognition model
- `openie_model`: Open Information Extraction model
- `num_processes`: Parallel processing threads
- `force`: Force recompute from scratch

#### STEP 2: KGConstructor Initialization
**Component**: `gfmrag.kg_construction.KGConstructor`

**Responsibilities**:
1. Initialize NER model for entity extraction
2. Initialize OpenIE model for relationship extraction
3. Configure processing parameters

**Models Used**:
- **NER**: Extracts named entities (Person, Organization, Location, etc.)
- **OpenIE**: Extracts (subject, relation, object) triples

#### STEP 3: Document Processing
**Process for each document**:

```
Document Text
     ↓
[NER] → Entities: {person, place, org, ...}
     ↓
[OpenIE] → Triples: (entity1, relation, entity2)
     ↓
[Validation] → Filter invalid triples
     ↓
[Storage] → Append to kg.txt
```

**Example**:
```
Input: "Aspirin treats headaches."

NER Output:
- "Aspirin" → drug
- "headaches" → symptom

OpenIE Output:
- (Aspirin, treats, headaches)

KG Triple:
aspirin,treats,headaches
```

#### STEP 4: Q&A Generation (Optional)
**Component**: `gfmrag.kg_construction.QAConstructor`

**Process**:
1. Read document content
2. Generate questions using LLM
3. Extract answers from text
4. Store Q&A pairs for retrieval

#### STEP 5: Knowledge Graph Storage
**Output Format**: `tmp/kg_construction/{dataset_name}/kg.txt`

```
entity1,relation,entity2
aspirin,treats,headache
diabetes,caused_by,insulin resistance
heart,is_part_of,cardiovascular system
```

**Format**:
- Delimiter: `,` (comma)
- Structure: `head,relation,tail`
- Encoding: UTF-8

---

## COMPONENTS

### 1. KGConstructor
**Purpose**: Build Knowledge Graph from documents

**Key Methods**:
- `from_config(cfg)`: Factory method to create instance
- `extract_entities(text)`: NER extraction
- `extract_relationships(text, entities)`: OpenIE extraction
- `validate_triple(head, rel, tail)`: Quality checks

**Configuration**:
```python
kg_constructor:
  root: tmp/kg_construction
  num_processes: 10
  add_title: True
  force: True
```

### 2. QAConstructor
**Purpose**: Generate Q&A pairs for RAG

**Key Methods**:
- `from_config(cfg)`: Factory method
- `generate_qa(document)`: Create Q&A pairs
- `store_pairs(qa_pairs)`: Save to database

**Note**: Optional component, can be disabled in config.

### 3. KGIndexer
**Purpose**: Orchestrate KG construction and QA generation

**Key Methods**:
- `index_data(dataset)`: Main entry point
- `process_document(doc)`: Process single document
- `save_graph()`: Persist Knowledge Graph

---

## CONFIGURATION

### Main Config File
`gfmrag/workflow/config/stage1_index_dataset.yaml`

```yaml
hydra:
  run:
    dir: outputs/kg_construction/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - ner_model: llm_ner_model       # NER model config
  - openie_model: llm_openie_model # OpenIE model config

dataset:
  root: ./data                      # Data root directory
  data_name: hotpotqa               # Dataset name

kg_constructor:
  _target_: gfmrag.kg_construction.KGConstructor
  open_ie_model: ${openie_model}
  ner_model: ${ner_model}
  el_model: null                    # Entity linking disabled (uses Stage 2)
  root: tmp/kg_construction         # Output directory
  num_processes: 10                 # Parallel threads
  cosine_sim_edges: False           # Disabled (replaced by Stage 2)
  add_title: True                   # Include title in OpenIE
  force: True                       # Always recompute
```

### Override Examples
```bash
# Change dataset
python -m gfmrag.workflow.stage1_index_dataset dataset.data_name=medqa

# Change number of processes
python -m gfmrag.workflow.stage1_index_dataset kg_constructor.num_processes=20

# Change NER model
python -m gfmrag.workflow.stage1_index_dataset ner_model=custom_ner
```

---

## INPUT/OUTPUT

### Input

**Format**: Dataset directory structure
```
data/
  hotpotqa/
    train.json
    dev.json
    test.json
```

**Dataset Schema** (HotpotQA example):
```json
{
  "id": "doc_001",
  "question": "What treats headaches?",
  "context": [
    ["title", ["Aspirin is a medication...", "It treats headaches..."]]
  ],
  "answer": "Aspirin"
}
```

### Output

#### 1. Knowledge Graph: `tmp/kg_construction/{dataset}/kg.txt`
```
Format: head,relation,tail
Example:
aspirin,treats,headache
aspirin,is_a,medication
headache,is_a,symptom
```

**Statistics** (typical for HotpotQA):
- Total triples: ~50,000
- Unique entities: ~10,000
- Unique relations: ~500

#### 2. Intermediate Files
```
tmp/kg_construction/{dataset}/
  ├─ kg.txt              # Final Knowledge Graph
  ├─ entities.txt        # All extracted entities
  ├─ relations.txt       # All unique relations
  └─ statistics.json     # Processing stats
```

---

## METRICS & EVALUATION

### Key Metrics

#### 1. **Extraction Quality**
- **Total Triples Extracted**: Number of raw triples from OpenIE
- **Clean Triples**: Valid triples after filtering
- **Unique Triples**: Deduplicated triples
- **Formatting Errors**: Malformed triples
- **NER Coverage**: Percentage of triples with NER entities

#### 2. **Entity Statistics**
- **Total Entities**: All entity mentions
- **Unique Entities**: Distinct entities
- **Entity Types**: Distribution across types (person, org, etc.)

#### 3. **Relationship Statistics**
- **Total Relations**: All relation mentions
- **Unique Relations**: Distinct relation types
- **Relation Distribution**: Frequency of each relation

#### 4. **Efficiency**
- **Processing Time**: Total time for dataset
- **Throughput**: Documents/second
- **Memory Usage**: Peak RAM consumption

### Evaluation Approach

**Quality Checks**:
1. **Triple Validation**: Ensure (head, relation, tail) structure
2. **Entity Coverage**: Check NER entity presence
3. **Relation Coherence**: Validate relation semantics
4. **Deduplication**: Remove duplicate triples

**Success Criteria**:
- Clean triple rate: ≥ 90%
- Entity coverage: ≥ 80%
- Unique triples: Deduplicated
- No formatting errors

---

## VISUALIZATIONS

### Currently Not Generated
Stage 1 does not generate visualizations by default.

### Recommended Visualizations
If you want to analyze Stage 1 output:

1. **Entity Type Distribution**
   - Bar chart of entity type counts
   - Pie chart of type percentages

2. **Relation Frequency**
   - Histogram of top N relations
   - Word cloud of relation types

3. **Triple Quality**
   - Stacked bar: clean vs malformed vs missing NER
   - Time series of extraction rate

4. **Processing Performance**
   - Bar chart: time per stage
   - Line plot: throughput over time

---

## USAGE

### Basic Usage

```bash
# Run Stage 1 with default config
python -m gfmrag.workflow.stage1_index_dataset
```

### Advanced Usage

```bash
# Override dataset
python -m gfmrag.workflow.stage1_index_dataset \
  dataset.data_name=medqa \
  dataset.root=./my_data

# Change processing parameters
python -m gfmrag.workflow.stage1_index_dataset \
  kg_constructor.num_processes=20 \
  kg_constructor.force=False

# Custom output directory
python -m gfmrag.workflow.stage1_index_dataset \
  kg_constructor.root=tmp/my_kg
```

### Programmatic Usage

```python
from gfmrag import KGIndexer
from gfmrag.kg_construction import KGConstructor, QAConstructor
from omegaconf import OmegaConf

# Load config
cfg = OmegaConf.load("gfmrag/workflow/config/stage1_index_dataset.yaml")

# Initialize constructors
kg_constructor = KGConstructor.from_config(cfg.kg_constructor)
qa_constructor = QAConstructor.from_config(cfg.qa_constructor)

# Create indexer
indexer = KGIndexer(kg_constructor, qa_constructor)

# Index data
indexer.index_data(cfg.dataset)
```

---

## PERFORMANCE

### Benchmark (HotpotQA dataset)

**Hardware**:
- CPU: 10 cores
- RAM: 32GB
- GPU: Not required for Stage 1

**Metrics**:
| Metric | Value |
|--------|-------|
| Documents | 90,447 |
| Total Triples | 52,384 |
| Unique Entities | 12,093 |
| Processing Time | ~6 hours |
| Throughput | ~4 docs/sec |
| Memory Usage | ~8GB peak |

### Optimization Tips

1. **Increase Parallelism**
   ```yaml
   kg_constructor.num_processes: 20  # More threads
   ```

2. **Disable QA Generation**
   ```yaml
   qa_constructor: null  # Skip if not needed
   ```

3. **Batch Processing**
   - Process documents in batches
   - Reduces memory overhead

4. **Model Optimization**
   - Use faster NER models
   - Simplify OpenIE extraction

---

## TROUBLESHOOTING

### Common Issues

#### 1. Out of Memory
**Symptom**: Process killed, "Out of memory" error

**Solution**:
```yaml
# Reduce parallel processes
kg_constructor.num_processes: 5

# Or process in batches
```

#### 2. Slow Processing
**Symptom**: Very low throughput

**Solution**:
- Check GPU availability for models
- Increase `num_processes`
- Profile bottlenecks

#### 3. Empty Output
**Symptom**: kg.txt is empty or very small

**Solution**:
- Check dataset path is correct
- Verify dataset format
- Enable verbose logging

#### 4. Format Errors
**Symptom**: Malformed triples in output

**Solution**:
- Check OpenIE model configuration
- Enable triple validation
- Review extraction logs

---

## NEXT STEPS

After Stage 1 completes:

1. **Verify Output**
   ```bash
   head -n 20 tmp/kg_construction/hotpotqa/kg.txt
   wc -l tmp/kg_construction/hotpotqa/kg.txt
   ```

2. **Proceed to Stage 2**
   ```bash
   python -m gfmrag.workflow.stage2_entity_resolution
   ```

3. **Review Statistics**
   - Check `statistics.json` for quality metrics
   - Ensure clean triple rate ≥ 90%

---

## REFERENCES

### Code Files
- Main: `gfmrag/workflow/stage1_index_dataset.py`
- Config: `gfmrag/workflow/config/stage1_index_dataset.yaml`
- KGConstructor: `gfmrag/kg_construction/__init__.py`
- QAConstructor: `gfmrag/kg_construction/__init__.py`

### Documentation
- Hydra Config: https://hydra.cc/docs/intro/
- OpenIE: https://github.com/dair-iitd/OpenIE-standalone
- Named Entity Recognition: https://spacy.io/usage/linguistic-features#named-entities

---

**Last Updated**: 2026-01-05
**Version**: 1.0
**Author**: GFM-RAG Team
