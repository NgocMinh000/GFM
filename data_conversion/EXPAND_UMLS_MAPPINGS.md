# Mở rộng UMLS Mappings cho các Sources khác

## 📊 Tình trạng hiện tại

**Đã map:**
- MONDO (536K entities) → 244K mapped → 32K triples ✅

**Chưa map:**
- DrugBank (5.6M) - Drugs
- NCBI (5.2M) - Genes/Proteins
- UBERON (3.1M) - Anatomy
- GO (884K) - Gene Ontology
- HPO (514K) - Phenotypes
- MONDO_grouped (145K)
- REACTOME (95K) - Pathways
- CTD (18K) - Chemical-Disease

**Mục tiêu:** Map tất cả sources sang UMLS CUIs để tăng từ 32K → 500K-2M triples

---

## 🎯 Strategies

### Strategy 1: Parse thêm .obo files (Giống MONDO)

Nhiều ontologies có UMLS cross-references trong .obo files:

#### A. HPO (Human Phenotype Ontology)

**Download:**
```bash
wget https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo
mv hp.obo primekg_data/hp.obo
```

**Parse tương tự MONDO:**
- Tìm `xref: UMLS:Cxxxxxxx`
- Tạo `hpo_to_umls.csv`
- Format: `hp_id,umls_id`

#### B. GO (Gene Ontology)

**Download:**
```bash
wget http://purl.obolibrary.org/obo/go.obo
mv go.obo primekg_data/go.obo
```

**Parse:**
- Tìm `xref: UMLS:` trong go.obo
- Tạo `go_to_umls.csv`

### Strategy 2: Sử dụng BioPortal Mappings

BioPortal cung cấp mappings giữa các ontologies:

```bash
# HPO to UMLS
curl "https://data.bioontology.org/ontologies/HP/mappings" > hpo_mappings.json

# GO to UMLS
curl "https://data.bioontology.org/ontologies/GO/mappings" > go_mappings.json
```

**Lưu ý:** Cần API key (free registration)

### Strategy 3: Sử dụng UMLS Metathesaurus

Nếu bạn có UMLS license (free for researchers):

1. Download UMLS Metathesaurus từ https://www.nlm.nih.gov/research/umls/
2. Extract MRCONSO.RRF file
3. Filter cho sources cần thiết:
   - SAB='DRUGBANK' (DrugBank)
   - SAB='HGNC' (NCBI genes)
   - SAB='HPO' (HPO)
   - SAB='GO' (Gene Ontology)

```bash
# Extract DrugBank IDs
grep "DRUGBANK" MRCONSO.RRF | cut -f1,14 | sort -u > drugbank_to_umls.csv

# Extract NCBI Gene IDs
grep "HGNC" MRCONSO.RRF | cut -f1,14 | sort -u > ncbi_to_umls.csv
```

### Strategy 4: Incremental Approach

Map từng source một theo priority:

**Priority 1: HPO** (514K phenotypes)
- Download hp.obo
- Parse xref: UMLS
- Expected: ~300K-400K mappings

**Priority 2: GO** (884K terms)
- Download go.obo
- Parse xref: UMLS
- Expected: ~200K-400K mappings

**Priority 3: DrugBank** (5.6M)
- Phức tạp hơn, cần RxNorm hoặc UMLS
- Có thể bỏ qua nếu không cần drug data

**Priority 4: NCBI Genes** (5.2M)
- Cần UMLS HGNC/Gene source
- Hoặc sử dụng MyGene.info API

---

## 🚀 Implementation Plan

### Bước 1: Chạy analysis script

```bash
git pull origin claude/analyze-stage3-umls-mapping-0cGgL
python analyze_mapping_opportunities.py
```

Xem sources nào có potential cao nhất.

### Bước 2: Parse HPO .obo file (Easiest)

Tôi sẽ tạo script tương tự `create_umls_mondo_mapping.py` nhưng cho HPO:

```bash
# Download HPO
wget https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo -O primekg_data/hp.obo

# Parse
python create_umls_hpo_mapping.py

# Output: primekg_data/hpo_to_umls.csv
```

### Bước 3: Parse GO .obo file

```bash
# Download GO
wget http://purl.obolibrary.org/obo/go.obo -O primekg_data/go.obo

# Parse
python create_umls_go_mapping.py

# Output: primekg_data/go_to_umls.csv
```

### Bước 4: Update converter để support multiple mappings

Modify `primekg_to_umls_triples.py` để load nhiều mapping files:

```python
self.mondo_to_umls = load_mapping('umls_mondo.csv')
self.hpo_to_umls = load_mapping('hpo_to_umls.csv')  # NEW
self.go_to_umls = load_mapping('go_to_umls.csv')    # NEW
```

### Bước 5: Re-run conversion

```bash
python primekg_pipeline.py --skip-download --strategy map_all
```

---

## 📈 Expected Results

**Với MONDO only (hiện tại):**
- 32,886 triples (0.4% of 8M)
- 11,424 unique entities

**Với MONDO + HPO:**
- ~300K-500K triples (3-6% of 8M)
- ~150K-250K unique entities

**Với MONDO + HPO + GO:**
- ~500K-800K triples (6-10% of 8M)
- ~250K-400K unique entities

**Với All sources (MONDO + HPO + GO + DrugBank + NCBI):**
- ~2M-4M triples (25-50% of 8M)
- ~800K-1.5M unique entities

---

## ⚠️ Challenges

### 1. DrugBank → UMLS

**Vấn đề:** DrugBank IDs không trực tiếp map sang UMLS
**Giải pháp:**
- Sử dụng RxNorm (UMLS source cho drugs)
- Hoặc DrugBank cung cấp mapping files (cần license)
- Hoặc bỏ qua drugs (focus on diseases/phenotypes/genes)

### 2. NCBI Gene → UMLS

**Vấn đề:** NCBI Gene IDs là integers (9796, 7918...)
**Giải pháp:**
- Sử dụng HGNC symbols làm intermediate
- Hoặc MyGene.info API để map
- Hoặc UMLS HGNC source

### 3. UBERON → UMLS

**Vấn đề:** Limited UMLS coverage for anatomy
**Giải pháp:**
- Check uberon.obo for xref: UMLS
- Hoặc sử dụng UMLS anatomy sources
- Có thể coverage thấp

---

## 💡 Recommendations

**Cho GFM-RAG project:**

1. **Start with diseases + phenotypes:**
   - MONDO (✅ done)
   - HPO (easy to add)
   - → Good coverage for medical use cases

2. **Add biological processes:**
   - GO (moderate difficulty)
   - → Useful for gene function

3. **Consider skipping:**
   - DrugBank (unless you need drug interactions)
   - NCBI Genes (complex mapping, may not need for diseases)
   - UBERON (low UMLS coverage)

**Khuyến nghị ngắn hạn:**
```bash
# Focus on HPO + MONDO
1. Parse hp.obo → hpo_to_umls.csv
2. Update converter
3. Re-run pipeline
4. Expect ~300K-500K triples (10x improvement!)
```

---

## 📞 Next Steps

Bạn muốn tôi:

**A. Implement HPO parser ngay** (giống MONDO parser)?
```bash
python create_umls_hpo_mapping.py
→ Quick win, ~10x triples
```

**B. Implement GO parser**?
```bash
python create_umls_go_mapping.py
→ More triples, biological processes
```

**C. Full solution với all sources**?
```bash
python create_umls_multi_mapping.py
→ Comprehensive, nhiều work hơn
```

**D. Analyze trước rồi quyết định**?
```bash
python analyze_mapping_opportunities.py
→ Xem potential của từng source
```

Bạn chọn phương án nào? Tôi sẽ implement ngay! 🚀
