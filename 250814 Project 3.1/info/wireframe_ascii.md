# Wireframe ASCII - Giao diện Topic Modeling Project (Wizard UI)

## 🎯 Tổng quan giao diện
Dự án Topic Modeling với giao diện **Wizard UI/Multi-step Form** theo kiến trúc **Progressive Disclosure**:
- **Step-by-step workflow** theo pipeline xử lý dữ liệu
- **Progressive Disclosure**: Chỉ hiển thị thông tin cần thiết ở mỗi bước
- **Streamlit-friendly**: Dễ lập trình với session state và containers
- **User Guidance**: Hướng dẫn rõ ràng cho từng bước

---

## 🧙‍♂️ **WIZARD UI LAYOUT - Multi-Step Form (6 Steps)**

### **STEP 1: Dataset Selection & Upload + Sampling Configuration**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 1/6: Dataset Selection & Upload                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🎯 Choose Dataset Source:                                               │   │
│  │     ○ File Path (File Path)                                             │   │
│  │     ○ Upload Custom File (CSV/JSON/Excel)                               │   │
│  │     ○ Use Sample Dataset (Cache Folder)                                 │   │
│  │                                                                         │   │
│  │  📁 If Custom Upload:                                                    │   │
│  │     [Choose Files] [Browse Files]                                       │   │
│  │                                                                         │   │
│  │  📊 Dataset Preview (if available):                                     │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ [Data Preview - First 5 rows]                               │     │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  📈 Dataset Info:                                                       │   │
│  │     • Shape: [X rows, Y columns]                                       │   │
│  │     • Memory: [Z MB]                                                   │   │
│  │     • Format: [CSV/JSON/Excel]                                         │   │
│  │                                                                         │   │
│  │  ─────────────────────────────────────────────────────────────────────   │   │
│  │                                                                         │   │
│  │  🔧 Sampling Configuration:                                              │   │
│  │     ┌───────────────────────────────┬───────────────────────────────┐    │   │
│  │     │  📊 Number of Samples:        │  🎯 Sampling Strategy:        │    │   │
│  │     │  [Slider: 1K - 500K]          │  ○ Random                     │    │   │
│  │     │                               │  ○ Stratified (Recommended)   │    │   │
│  │     └───────────────────────────────┴───────────────────────────────┘    │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [◀ Previous] [Next ▶]                                                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 2: Column Selection & Preprocessing**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 2/6: Column Selection & Preprocessing                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  📝 Column Selection:                                                    │   │
│  │                                                                         │   │
│  │     ┌───────────────────────────────┬───────────────────────────────┐    │   │
│  │     │  📄 Chọn cột văn bản:         │  🏷️  Chọn cột nhãn:           │   │
│  │     │  [Chọn từ danh sách ▼]        │  [Chọn từ danh sách ▼]        │   │
│  │     └───────────────────────────────┴───────────────────────────────┘    │   │
│  │                                                                         │   │
│  │  ─────────────────────────────────────────────────────────────────────   │   │
│  │                                                                         │   │
│  │  🔍 Column Analysis:                                                     │   │
│  │                                                                         │   │
│  │     ┌───────────────────────────────┬───────────────────────────────┐    │   │
│  │     │      Text Column              │        Label Column           │    │   │
│  │     ├───────────────────────────────┼───────────────────────────────┤    │   │
│  │     │ [Column Name]                 │ [Column Name]                 │    │   │
│  │     │ • Samples: [X]                │ • Unique Classes: [X]         │    │   │
│  │     │ • Avg Length: [Y chars]       │ • Distribution: [Balanced/    │    │   │
│  │     │ • Unique Words: [Z]           │   Imbalanced]                 │    │   │
│  │     │                               │ • Sample Labels: [Class1,     │    │   │
│  │     │                               │   Class2, ...]                │    │   │
│  │     └───────────────────────────────┴───────────────────────────────┘    │   │
│  │                                                                         │   │
│  │  ─────────────────────────────────────────────────────────────────────   │   │
│  │                                                                         │   │
│  │  ✅ Column Validation:                                                   │   │
│  │     • Validation Errors: [List of errors if any]                       │   │
│  │     • Validation Warnings: [List of warnings if any]                   │   │
│  │     • Status: [All validations passed!]                                │   │
│  │                                                                         │   │
│  │  ─────────────────────────────────────────────────────────────────────   │   │
│  │                                                                         │   │
│  │  🧹 Preprocessing Options:                                               │   │
│  │     ┌───────────────────────────────┬───────────────────────────────┐    │   │
│  │     │  ☑️ Text Cleaning              │  ☑️ Data Validation           │    │   │
│  │     │  (remove special chars)       │  (remove nulls)               │   │
│  │     │                               │                               │   │
│  │     │  ☑️ Category Mapping           │  ☑️ Memory Optimization       │   │
│  │     │  (convert to numeric)         │                               │   │
│  │     └───────────────────────────────┴───────────────────────────────┘    │   │
│  │                                                                         │   │
│  │  ─────────────────────────────────────────────────────────────────────   │   │
│  │                                                                         │   │
│  │  👀 Column Preview:                                                      │   │
│  │     [Dataframe preview of selected columns - First 10 rows]             │   │
│  │                                                                         │   │
│  │  [💾 Save Column Configuration]                                          │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [◀ Previous] [Next ▶]                                                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 3: Model Configuration & Vectorization**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 3/6: Model Configuration & Vectorization                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🔧 Manual Settings (if selected):                                       │   │
│  │     📊 Data Split:                                                       │   │
│  │        • Training: [60%] [Slider]                                       │   │
│  │        • Validation: [20%] [Slider]                                     │   │
│  │        • Test: [20%] [Slider]                                           │   │
│  │                                                                         │   │
│  │     🔄 Cross-Validation:                                                 │   │
│  │        • CV Folds: [5] [Slider: 3-10]                                  │   │
│  │        • Random State: [42] [Input]                                     │   │
│  │                                                                         │   │
│  │  🎯 Model Selection:                                                     │   │
│  │     ☑️ K-Means Clustering (Unsupervised)                               │   │
│  │     ☑️ K-Nearest Neighbors (Supervised)                                │   │
│  │     ☑️ Decision Tree (Supervised)                                       │   │
│  │     ☑️ Naive Bayes (Supervised)                                         │   │
│  │                                                                         │   │
│  │  📚 Text Vectorization Methods:                                          │   │
│  │     ☑️ Bag of Words (BoW) - Fast, interpretable                        │   │
│  │     ☑️ TF-IDF - Better than BoW, handles rare words                    │   │
│  │     ☑️ Word Embeddings - Semantic understanding, slower                │   │
│  │                                                                         │   │
│  │                                                                        │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│ [◀ Previous]  [🚀 Start Training]                                              |
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 4: Training Execution & Monitoring**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 4/6: Training Execution & Monitoring                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🚀 Training Control:                                                    │   │
│  │     [START TRAINING] [PAUSE] [STOP] [RESET]                             │   │
│  │                                                                         │   │
│  │  📊 Overall Progress:                                                    │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ [██████████] 100% - Training Complete!                      │     │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  ⏱️  Time Information:                                                   │   │
│  │     • Elapsed Time: [X minutes]                                         │   │
│  │     • Estimated Remaining: [Y minutes]                                  │   │
│  │     • Total Expected: [Z minutes]                                       │   │
│  │                                                                         │   │
│  │  🔄 Current Step:                                                        │   │
│  │     • Phase: [Model Training]                                           │   │
│  │     • Model: [K-Means + BoW]                                           │   │
│  │     • Status: [Training...]                                             │   │
│  │                                                                         │   │
│  │  📈 Real-time Metrics:                                                   │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ Live Training Progress:                                        │   │
│  │     │ • Models Completed: [3/12]                                     │   │
│  │     │ • Current Accuracy: [XX.XX%]                                   │   │
│  │     │ • Best So Far: [XX.XX%]                                        │   │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 5: Results Analysis & Export**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 5/6: Results Analysis & Export                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🏆 Best Model Selection:                                                │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ 🥇 TOP PERFORMER: [Model Name + Vectorization]             │     │
│  │     │ • Accuracy: [XX.XX%]                                         │     │
│  │     │ • Training Time: [X minutes]                                 │     │
│  │     │ • Memory Usage: [Y MB]                                       │     │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  📊 Performance Summary:                                                 │   │
│  │     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │   │
│  │     │   Accuracy      │ │   Precision     │ │   Recall        │       │   │
│  │     │   [XX.XX%]      │ │   [XX.XX%]      │ │   [XX.XX%]      │       │   │
│  │     └─────────────────┘ └─────────────────┘ └─────────────────┘       │   │
│  │                                                                         │   │
│  │  🎯 Model Comparison Chart:                                              │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ [Interactive Bar Chart: Model vs Accuracy]                  │     │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  📁 Export Options:                                                     │   │
│  │     [📊 CSV Report] [📈 PDF Charts]                                   │   │
│  │                                                                         │   │
│  │  🔍 Detailed Model Analysis:                                            │   │
│  │     [📊 View All Model Results] - Detailed evaluation for each model   │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [◀ Previous] [Next ▶]                                                      │   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 6: Text Classification & Inference**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 6/6: Text Classification & Inference                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🎯 **Text Input & Classification**                                       │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ 📝 Enter your text for classification:                        │     │
│  │     │ ┌─────────────────────────────────────────────────────────┐ │     │   │
│  │     │ │ [Text input area - Multi-line support]                  │ │     │   │
│  │     │ │ [Placeholder: "Enter text to classify..."]              │ │     │   │
│  │     │ └─────────────────────────────────────────────────────────┘ │     │   │
│  │     │                                                             │     │   │
│  │     │ [🔍 Classify Text] [📋 Batch Upload] [🔄 Clear]            │     │   │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  🏆 **Classification Results**                                           │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ 🎯 **Predicted Class**: [Class Name]                           │     │
│  │     │ 📊 **Confidence Score**: [XX.XX%]                              │     │
│  │     │ 🏷️  **Top 3 Predictions**:                                    │     │
│  │     │    • [Class 1]: [XX.XX%]                                       │     │
│  │     │    • [Class 2]: [XX.XX%]                                       │     │
│  │     │    • [Class 3]: [XX.XX%]                                       │     │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  📈 **Model Insights**                                                  │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ 🔍 **Key Features Used**:                                      │     │
│  │     │    • [Feature 1]: [Weight]                                     │     │
│  │     │    • [Feature 2]: [Weight]                                     │     │
│  │     │    • [Feature 3]: [Weight]                                     │     │
│  │     │                                                             │     │
│  │     │ 📊 **Text Analysis**:                                           │     │
│  │     │    • Word count: [X]                                            │     │
│  │     │    • Unique words: [Y]                                          │     │
│  │     │    • Processing time: [Z ms]                                    │     │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  🔽 **Advanced Analysis** (Click to expand)                            │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ 📊 **Detailed Probabilities**: [Bar chart of all classes]      │     │
│  │     │ 🔍 **Feature Importance**: [Word cloud/feature weights]        │     │
│  │     │ 📈 **Model Confidence**: [Confidence distribution]              │     │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  📁 **Export & Save**                                                  │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ [💾 Save Classification] [📊 Export Results] [📋 Add to Batch] │     │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [◀ Previous] [🏠 Start Over] [💾 Save Results]                              │   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 **SIDEBAR - Progress Tracker & Quick Actions**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 📋 PROGRESS TRACKER                                                            │
│                                                                                 │
│ 🎯 Current Step: [2/6] - Column Selection & Preprocessing                     │
│                                                                                 │
│ 📍 Step Status:                                                                │
│   ✅ Step 1: Dataset Selection (Completed)                                    │
│   🔄 Step 2: Column Selection & Preprocessing (Current)                       │
│   ⏳ Step 3: Model Configuration (Pending)                                    │
│   ⏳ Step 4: Training Execution (Pending)                                     │
│   ⏳ Step 5: Results Analysis (Pending)                                       │
│   ⏳ Step 6: Text Classification (Pending)                                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ 📊 QUICK STATS                                                                │
│                                                                                 │
│ Dataset Size: [X MB]                                                          │
│ Samples Selected: [Y]                                                          │
│ Columns Available: [Z]                                                         │
│ Estimated Time: [W min]                                                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ 🎯 QUICK ACTIONS                                                              │
│                                                                                 │
│ [🔄 Reset Current Step]                                                       │
│ [📥 Load Preset Config]                                                       │
│ [💾 Save Progress]                                                            │
│ [📤 Export Current State]                                                     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│ 📱 VIEW CONTROLS                                                               │
│                                                                                 │
│ [📱 Compact View]                                                             │
│ [💻 Detailed View]                                                            │
│ [🎨 Theme Toggle]                                                             │
│ [🔍 Help & Tips]                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📱 **MOBILE RESPONSIVE - Compact Wizard**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling                                                              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP [2/6]: Column Selection & Preprocessing                              │
│                                                                                 │
│  📝 Text Column: [Dropdown ▼]                                                 │
│  🏷️  Label Column: [Dropdown ▼]                                               │
│                                                                                 │
│  📊 Quick Stats: [X samples, Y classes]                                       │
│                                                                                 │
│  [◀ Previous] [Next ▶]                                                        │
│                                                                                 │
│  📍 Progress: ████████░░ 75%                                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎨 **COLOR SCHEME & STYLING - Wizard Theme**

```
Primary Colors:
- Header: #0d5f3c (Deep Royal Green)
- Step Indicator: #16a085 (Emerald)
- Success: #27ae60 (Success Green)
- Info: #48c9b0 (Aqua Green)
- Warning: #f39c12 (Golden Orange)
- Error: #e74c3c (Coral Red)

Wizard Styling:
- Step Header: #ecf8f5 (Mint Cream)
- Active Step: #d5f4e6 (Soft Mint)
- Completed Step: #a9dfbf (Light Emerald)
- Pending Step: #eafaf1 (Whisper Green)

Typography:
- Step Title: 1.5rem, Bold, #0d5f3c
- Step Description: 1rem, Normal, #5d6d5b
- Form Labels: 1rem, Semi-bold, #34495e
- Help Text: 0.875rem, Light, #7b8471
```

---

## 🔄 **USER FLOW - Wizard Navigation (6 Steps)**

```
1. User lands on Step 1 (Dataset Selection + Sampling Configuration)
   ↓
2. Completes Step 1 → Auto-advance to Step 2
   ↓
3. Completes Step 2 (Column Selection + Preprocessing) → Auto-advance to Step 3
   ↓
4. Completes Step 3 (Model Configuration) → Auto-advance to Step 4
   ↓
5. Completes Step 4 (Training Execution) → Auto-advance to Step 5
   ↓
6. Completes Step 5 (Results Analysis) → Auto-advance to Step 6
   ↓
7. Completes Step 6 (Text Classification) → Results & Export
   ↓
8. Option to Start Over or Save Results
```

---

## 📱 **RESPONSIVE BREAKPOINTS - Wizard Layout**

```
Desktop: ≥ 1200px (Full wizard with sidebar)
Tablet: 768px - 1199px (Wizard with collapsible sidebar)
Mobile: < 768px (Compact wizard, no sidebar)
```

---

## 🎯 **KEY FEATURES - Wizard UI (Updated)**

```
✅ Step-by-Step Workflow (6 Steps)
✅ Progressive Disclosure
✅ Auto-advance on Completion
✅ Progress Tracking
✅ Validation at Each Step
✅ Skip to End Option
✅ Mobile Responsive
✅ Session State Management
✅ Error Handling & Recovery
✅ Configuration Persistence
✅ Sampling Configuration in Step 1
✅ Preprocessing Options in Step 2
✅ Column Selection & Validation
✅ Model Configuration & Vectorization
✅ Training Execution & Monitoring
✅ Results Analysis & Export
✅ Text Classification & Inference
```

---

## 🔧 **TECHNICAL IMPLEMENTATION - Streamlit**

### **Session State Structure:**
```python
# Streamlit session state for wizard
if 'wizard_step' not in st.session_state:
    st.session_state.wizard_step = 1

if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = {}

if 'sampling_config' not in st.session_state:
    st.session_state.sampling_config = {}

if 'step2_config' not in st.session_state:
    st.session_state.step2_config = {}

if 'model_config' not in st.session_state:
    st.session_state.model_config = {}

if 'training_results' not in st.session_state:
    st.session_state.training_results = {}
```

### **Wizard Step Management:**
```python
def next_step():
    if st.session_state.wizard_step < 6:
        st.session_state.wizard_step += 1
        st.rerun()

def previous_step():
    if st.session_state.wizard_step > 1:
        st.session_state.wizard_step -= 1
        st.rerun()

def go_to_step(step_number):
    if 1 <= step_number <= 6:
        st.session_state.wizard_step = step_number
        st.rerun()
```

### **Step Validation:**
```python
def validate_step(step):
    if step == 1:
        return 'dataset_info' in st.session_state and st.session_state.dataset_info
    elif step == 2:
        return 'step2_config' in st.session_state and st.session_state.step2_config.get('completed', False)
    elif step == 3:
        return 'model_config' in st.session_state and st.session_state.model_config.get('completed', False)
    elif step == 4:
        return 'training_complete' in st.session_state
    elif step == 5:
        return 'results_available' in st.session_state
    elif step == 6:
        return 'model_loaded' in st.session_state
    return False
```

---

## 🚀 **ADVANTAGES OF WIZARD UI**

### **User Experience:**
1. **Guided Workflow**: Users follow clear, logical steps
2. **Reduced Cognitive Load**: Only relevant options shown at each step
3. **Progress Visibility**: Clear indication of completion status
4. **Error Prevention**: Validation at each step prevents mistakes

### **Development Benefits:**
1. **Modular Code**: Each step is a separate function/component
2. **Easy Testing**: Test individual steps independently
3. **Maintainability**: Clear separation of concerns
4. **Scalability**: Easy to add new steps or modify existing ones

### **Streamlit Integration:**
1. **Session State**: Perfect for maintaining wizard state
2. **Container Management**: Easy to show/hide content based on step
3. **Form Validation**: Built-in form validation capabilities
4. **Responsive Design**: Automatic responsive behavior

---

## 🔄 **UPDATED STEP STRUCTURE - Current Implementation**

### **Step 1: Dataset Selection & Upload + Sampling Configuration**
- Dataset source selection (File Path, Upload, Sample Dataset)
- File upload and preview
- **Sampling Configuration** (moved from old Step 2)
  - Number of samples slider
  - Sampling strategy (Random/Stratified)

### **Step 2: Column Selection & Preprocessing** (formerly Step 3)
- Text and label column selection
- Column analysis and statistics
- Column validation
- **Preprocessing Options** (moved from old Step 2)
  - Text cleaning
  - Category mapping
  - Data validation
  - Memory optimization

### **Step 3: Model Configuration & Vectorization** (formerly Step 4)
- Model selection
- Text vectorization methods
- Training parameters

### **Step 4: Training Execution & Monitoring** (formerly Step 5)
- Training control
- Progress monitoring
- Real-time metrics

### **Step 5: Results Analysis & Export** (formerly Step 6)
- Model comparison
- Performance metrics
- Export options

### **Step 6: Text Classification & Inference** (formerly Step 7)
- Text input
- Classification results
- Model insights

---

*Giao diện Wizard UI này được thiết kế để tương thích hoàn toàn với Streamlit, dễ lập trình và cung cấp trải nghiệm người dùng tốt nhất theo nguyên tắc Progressive Disclosure. Cấu trúc đã được cập nhật từ 7 steps xuống 6 steps với việc tích hợp Sampling Configuration vào Step 1 và Preprocessing Options vào Step 2.*

### **MODEL DETAILED ANALYSIS WINDOW**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🔍 Detailed Model Analysis - All Evaluated Models                             │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📊 **Model Performance Overview**                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🎯 **Select Model for Detailed Analysis**:                              │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │   │
│  │  │ 🧠 K-Means      │ │ 🧠 KNN          │ │ 🧠 Decision     │           │   │
│  │  │ + BoW           │ │ + TF-IDF        │ │ Tree + Embed.   │           │   │
│  │  │ Accuracy: 85.2%│ │ Accuracy: 87.1%│ │ Accuracy: 89.3%│           │   │
│  │  │ [View Details] │ │ [View Details] │ │ [View Details] │           │   │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘           │   │
│  │                                                                         │   │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │   │
│  │  │ 🧠 Naive Bayes │ │ 🧠 SVM          │ │ 🧠 K-Means      │           │   │
│  │  │ + BoW           │ │ + TF-IDF        │ │ + Embeddings   │           │   │
│  │  │ Accuracy: 82.7%│ │ Accuracy: 88.9%│ │ Accuracy: 86.5%│           │   │
│  │  │ [View Details] │ │ [View Details] │ │ [View Details] │           │   │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘           │   │
│  │                                                                         │   │
│  │  📈 **Performance Metrics Summary**:                                     │   │
│  │     • Best Overall: Decision Tree + Embeddings (89.3%)                 │   │
│  │     • Fastest Training: K-Means + BoW (2.3 min)                       │   │
│  │     • Most Memory Efficient: Naive Bayes + BoW (45 MB)                │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [🔙 Back to Results] [📊 Export All Results] [🔄 Refresh]                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **INDIVIDUAL MODEL CONFUSION MATRIX WINDOW**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🧠 Model: Decision Tree + Word Embeddings - Detailed Analysis                 │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📊 **Confusion Matrix & Performance Metrics**                                │   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🎯 **Confusion Matrix**:                                               │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │                    Predicted                                │     │   │
│  │     │         ┌─────────┬─────────┬─────────┬─────────┐           │     │   │
│  │     │         │ Class 1 │ Class 2 │ Class 3 │ Class 4 │           │     │   │
│  │     │ ┌───────┼─────────┼─────────┼─────────┼─────────┤           │     │   │
│  │     │ │Class 1│   156   │    8    │    3    │    1    │           │     │   │
│  │     │ ├───────┼─────────┼─────────┼─────────┼─────────┤           │     │   │
│  │     │ │Class 2│    5    │   142   │    6    │    2    │           │     │   │
│  │     │ ├───────┼─────────┼─────────┼─────────┼─────────┤           │     │   │
│  │     │ │Class 3│    2    │    4    │   148   │    3    │           │     │   │
│  │     │ ├───────┼─────────┼─────────┼─────────┼─────────┤           │     │   │
│  │     │ │Class 4│    1    │    2    │    4    │   145   │           │     │   │
│  │     │ └───────┴─────────┴─────────┴─────────┴─────────┘           │     │   │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  📈 **Detailed Metrics**:                                                │   │
│  │     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │   │
│  │     │   Accuracy      │ │   Precision     │ │   Recall        │       │   │
│  │     │   89.3%         │ │   89.1%         │ │   89.5%         │       │   │
│  │     └─────────────────┘ └─────────────────┘ └─────────────────┘       │   │
│  │                                                                         │   │
│  │  🎯 **Per-Class Performance**:                                           │   │
│  │     • Class 1: Precision=92.9%, Recall=92.9%, F1=92.9%               │   │
│  │     • Class 2: Precision=90.4%, Recall=91.6%, F1=91.0%               │   │
│  │     • Class 3: Precision=91.9%, Recall=94.3%, F1=93.1%               │   │
│  │     • Class 4: Precision=96.0%, Recall=95.4%, F1=95.7%               │   │
│  │                                                                         │   │
│  │  🔍 **Model Insights**:                                                  │   │
│  │     • Most Confused Classes: Class 2 ↔ Class 3 (6 misclassifications) │   │
│  │     • Best Performing: Class 4 (96.0% precision)                      │   │
│  │     • Training Time: 4.2 minutes                                      │   │
│  │     • Memory Usage: 78 MB                                             │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  📁 **Export Options**:                                                       │
│     [📊 Export Confusion Matrix] [📈 Export Metrics] [📋 Full Report]        │
│                                                                                 │
│  [🔙 Back to Model List] [🏠 Back to Results] [🔄 Compare with Other Models] │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```
