# 📚 Documentation Index

Complete documentation for the Bolt Hole Detection System.

## 🚀 Getting Started

**New to the project?** Start here:

1. **[README.md](README.md)** *(~5 min read)*
   - Project overview
   - Quick feature summary
   - Installation instructions

2. **[QUICKSTART.md](QUICKSTART.md)** *(~10 min read)*
   - 3-step setup guide
   - Command-line examples
   - Python API usage

---

## 👨‍💻 Developer Documentation

**Building or extending the system?** Read these:

1. **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** *(~45 min read)* ⭐ START HERE
   - **System Architecture** — High-level overview and data flow
   - **Core Components** — Detailed breakdown of each module:
     - `app.py` — Streamlit dashboard and processing loop
     - `detector.py` — Core detection algorithm with step-by-step explanation
     - `colour_rules.py` — Colour mask extraction
     - `panel_finder.py` — ROI detection with text anchor
     - `tracker.py` — Persistent hole numbering
     - `utils.py` — Visualization and export utilities
   - **Algorithm Design** — Why horizontal projection? Why multi-colour verification?
   - **Performance Optimization** — Speed strategies and benchmarks
   - **Development Workflow** — Local setup, testing, contribution guidelines
   - **Troubleshooting** — Common issues and solutions

2. **[ARCHITECTURE.md](ARCHITECTURE.md)** *(~20 min read)* ⭐ VISUAL REFERENCE
   - **Component Diagram** — System relationships
   - **Data Flow Diagram** — Step-by-step processing pipeline
   - **Algorithm Flowchart** — Decision tree and logic flow
   - **Class & Function Structure** — Code organization
   - **Interface Contracts** — API signatures and data formats
   - **State Diagram** — App lifecycle states
   - **Dependency Graph** — Module dependencies
   - **Parameter Sensitivity** — Tuning guide
   - **Performance Benchmarks** — Speed comparisons

3. **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** *(if exists)*
   - Implementation notes
   - Known limitations

---

## 🔍 Quick Reference

### For Different Use Cases

| I want to... | Read this | Time |
|--|--|--|
| **Use the system** | QUICKSTART.md | 10 min |
| **Understand the algorithm** | DEVELOPER_GUIDE.md (Algorithm Design section) | 15 min |
| **Add a new feature** | DEVELOPER_GUIDE.md (Contributing Guidelines) | 20 min |
| **Optimize performance** | DEVELOPER_GUIDE.md (Performance Optimization) | 10 min |
| **Debug a problem** | DEVELOPER_GUIDE.md (Troubleshooting) | 5-15 min |
| **See the architecture** | ARCHITECTURE.md | 20 min |
| **Understand data flow** | ARCHITECTURE.md (Data Flow Diagram) | 15 min |

---

## 📋 Documentation Overview

### DEVELOPER_GUIDE.md (1925 lines)
```
├─ System Architecture (overview)
├─ Core Components (6 modules detailed)
│  ├─ app.py (Streamlit UI & main loop)
│  ├─ detector.py (Core algorithm with 7 steps)
│  ├─ colour_rules.py (Colour extraction)
│  ├─ panel_finder.py (ROI detection)
│  ├─ tracker.py (Hole numbering)
│  └─ utils.py (Visualization)
├─ Algorithm Design (design decisions explained)
├─ Performance Optimization (5 speedup strategies)
├─ Development Workflow (setup, testing)
├─ Testing & Validation (quality metrics)
├─ Troubleshooting (common issues)
└─ Contributing Guidelines (how to contribute)
```

### ARCHITECTURE.md (900+ lines)
```
├─ Component Diagram (visual relationships)
├─ Data Flow Diagram (processing pipeline)
├─ Class & Function Structure (code organization)
├─ Algorithm Flowchart (decision tree)
├─ Interface Contracts (API specs)
├─ State Diagram (app lifecycle)
├─ Dependency Graph (module dependencies)
├─ Parameter Sensitivity (tuning reference)
└─ Performance Benchmarks (speed measurements)
```

---

## 🎯 Key Topics Fast Links

### Understanding the Algorithm

1. **Horizontal Projection** 
   - DEVELOPER_GUIDE.md → "Design Decision: Horizontal Projection"
   
2. **Multi-Colour Verification**
   - DEVELOPER_GUIDE.md → "Design Decision: Multi-Colour Verification"
   
3. **Detection Pipeline (7 Steps)**
   - DEVELOPER_GUIDE.md → "detector.py" → "Key Functions" → "detect()"
   - ARCHITECTURE.md → "Data Flow Diagram"

### Optimization Strategies

1. **Frame Skip**
   - DEVELOPER_GUIDE.md → "Performance Optimization" → "Profiling Results"

2. **Resolution Scaling**
   - app.py (see `scale_factor` variable)

3. **ROI Caching**
   - DEVELOPER_GUIDE.md → "panel_finder.py" → "Why ROI Caching?"

4. **Headless Mode**
   - DEVELOPER_GUIDE.md → "Performance Controls" table

### Component Details

1. **Colour Extraction**
   - DEVELOPER_GUIDE.md → "colour_rules.py"

2. **Hole Tracking**
   - DEVELOPER_GUIDE.md → "tracker.py"

3. **ROI Detection**
   - DEVELOPER_GUIDE.md → "panel_finder.py"

---

## 🔧 Development Commands

```bash
# Setup
cd bolt
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
source .venv/bin/activate             # Mac/Linux
pip install -r requirements.txt

# Run
streamlit run app.py                  # Web UI
python test_detector.py sample.png    # CLI

# Validate
python -m py_compile detector.py      # Syntax check
pytest test_detector.py -v            # Unit tests

# Contribute
git checkout -b feature/my-feature
# ... make changes ...
git add .
git commit -m "feat: description"
git push origin feature/my-feature
# ... create Pull Request ...
```

---

## 📖 Documentation Standards

All documentation follows these conventions:

- **Code snippets** are complete and runnable
- **Diagrams** use ASCII art for universal readability
- **Tables** present complex information clearly
- **Sections** use clear hierarchy (H1, H2, H3)
- **Examples** show real usage patterns
- **Links** point to relevant sections

---

## 🤝 Contributing to Documentation

Found an error or want to improve docs?

1. Fork the repository
2. Create a branch: `git checkout -b docs/improvement`
3. Edit the `.md` file
4. Commit: `git commit -m "docs: improve clarity in algorithm section"`
5. Push: `git push origin docs/improvement`
6. Create a Pull Request

---

## 📞 Need Help?

- **For algorithm questions** → DEVELOPER_GUIDE.md (Algorithm Design)
- **For debugging** → DEVELOPER_GUIDE.md (Troubleshooting)
- **For architecture questions** → ARCHITECTURE.md
- **For getting started** → QUICKSTART.md

---

## 📊 Documentation Statistics

| Document | Lines | Topics | Code Examples | Diagrams |
|----------|-------|--------|----------------|----------|
| DEVELOPER_GUIDE.md | 1925 | 30+ | 15+ | 10+ |
| ARCHITECTURE.md | 900+ | 12+ | 8+ | 6+ |
| QUICKSTART.md | 50+ | 3 | 3 | 0 |
| This Index | - | - | - | - |

---

**Last Updated:** 2026-05-26  
**Maintained By:** Divya Upadhyay  
**Status:** Complete & Ready for Contribution
