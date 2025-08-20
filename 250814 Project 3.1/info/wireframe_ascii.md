# Wireframe ASCII - Giao diện Topic Modeling Project (Wizard UI)

## 🎯 Tổng quan giao diện
Dự án Topic Modeling với giao diện **Wizard UI/Multi-step Form** theo kiến trúc **Progressive Disclosure**:
- **Step-by-step workflow** theo pipeline xử lý dữ liệu
- **Progressive Disclosure**: Chỉ hiển thị thông tin cần thiết ở mỗi bước
- **Streamlit-friendly**: Dễ lập trình với session state và containers
- **User Guidance**: Hướng dẫn rõ ràng cho từng bước

---

## 🧙‍♂️ **WIZARD UI LAYOUT - Multi-Step Form**

### **STEP 1: Dataset Selection & Upload**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 1/7: Dataset Selection & Upload                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🎯 Choose Dataset Source:                                               │   │
│  │     ○ HuggingFace (Recommended) - ArXiv Abstracts                       │   │
│  │     ○ Upload Custom File (CSV/JSON/Excel)                               │   │
│     ○ Use Sample Dataset (Demo)                                         │   │
│  │                                                                         │   │
│  │  📁 If Custom Upload:                                                    │   │
│  │     [Choose Files] [Browse Files]                                       │   │
│  │                                                                         │   │
│  │  📊 Dataset Preview (if available):                                     │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ [Data Preview - First 5 rows]                               │     │   │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  📈 Dataset Info:                                                       │   │
│  │     • Shape: [X rows, Y columns]                                       │   │
│  │     • Memory: [Z MB]                                                   │   │
│  │     • Format: [CSV/JSON/Excel]                                         │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [◀ Previous] [Next ▶] [Skip to End]                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 2: Data Preprocessing & Sampling**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 2/7: Data Preprocessing & Sampling                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🔧 Sampling Configuration:                                              │   │
│  │     📊 Number of Samples: [100,000] [Slider: 1K - 500K]                │   │
│  │     🎯 Sampling Strategy: ○ Random ○ Stratified (Recommended)          │   │
│  │                                                                         │   │
│  │  🧹 Preprocessing Options:                                               │   │
│  │     ☑️ Text Cleaning (remove special chars)                             │   │
│  │     ☑️ Category Mapping (convert to numeric)                            │   │
│  │     ☑️ Data Validation (remove nulls)                                  │   │
│  │     ☑️ Memory Optimization                                              │   │
│  │                                                                         │   │
│  │  📊 Current Dataset Status:                                              │   │
│  │     • Original Size: [X rows]                                           │   │
│  │     • After Sampling: [Y rows]                                          │   │
│  │     • Categories: [Z unique classes]                                    │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [◀ Previous] [Next ▶] [Skip to End]                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 3: Column Selection & Validation**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 3/7: Column Selection & Validation                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  📝 Column Selection:                                                    │   │
│  │     📄 Text Column: [Dropdown ▼] (Select column containing text)       │   │
│  │     🏷️  Label Column: [Dropdown ▼] (Select column containing labels)   │   │
│  │                                                                         │   │
│  │  🔍 Column Analysis:                                                     │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ Text Column: [Column Name]                                  │     │   │
│  │     │ • Samples: [X]                                              │     │   │
│  │     │ • Avg Length: [Y words]                                     │     │   │
│  │     │ • Unique Words: [Z]                                         │     │   │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ Label Column: [Column Name]                                 │     │   │
│  │     │ • Unique Classes: [X]                                       │   │
│  │     │ • Distribution: [Balanced/Imbalanced]                       │   │
│  │     │ • Sample Labels: [Class1, Class2, ...]                     │   │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  ✅ Validation Status:                                                   │   │
│  │     • Text Column: ✅ Valid (contains text data)                       │   │
│  │     • Label Column: ✅ Valid (contains categorical data)               │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [◀ Previous] [Next ▶] [Skip to End]                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 4: Model Configuration & Vectorization**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 4/6: Model Configuration & Vectorization                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  ⚙️  Configuration Mode:                                                  │   │
│  │     ○ Auto Configuration (Recommended) - Let AI choose best settings    │   │
│  │     ○ Manual Configuration - Customize all parameters                   │   │
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
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [◀ Previous] [Next ▶] [Skip to End]                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 5: Training Execution & Monitoring**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 5/6: Training Execution & Monitoring                                 │
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
│  [◀ Previous] [Next ▶] [Skip to End]                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 6: Results Analysis & Export**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 6/6: Results Analysis & Export                                       │
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
│  │     │ [Interactive Bar Chart: Model vs Accuracy]                  │     │   │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  📁 Export Options:                                                     │   │
│  │     [📊 CSV Report] [📈 PDF Charts] [🎯 Model Files] [📋 Summary]     │   │
│  │                                                                         │
│  │  🔍 Detailed Model Analysis:                                            │
│  │     [📊 View All Model Results] - Detailed evaluation for each model   │
│  │                                                                         │
│  │  🔄 Next Steps:                                                          │
│  │     • [🔄 Run Again] - Test with different parameters                 │
│  │     • [💾 Save Configuration] - Save current settings                 │
│  │     • [📤 Deploy Model] - Export for production use                   │
│  │                                                                         │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [◀ Previous] [🏠 Start Over] [💾 Save Results]                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **STEP 7: Text Classification & Inference**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 Topic Modeling - Auto Classifier                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 STEP 7/7: Text Classification & Inference                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  🎯 **Text Input & Classification**                                       │
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
│  │  🏆 **Classification Results**                                           │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ 🎯 **Predicted Class**: [Class Name]                           │     │   │
│  │     │ 📊 **Confidence Score**: [XX.XX%]                              │     │   │
│  │     │ 🏷️  **Top 3 Predictions**:                                    │     │   │
│  │     │    • [Class 1]: [XX.XX%]                                       │     │   │
│  │     │    • [Class 2]: [XX.XX%]                                       │     │   │
│  │     │    • [Class 3]: [XX.XX%]                                       │     │   │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  📈 **Model Insights**                                                  │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ 🔍 **Key Features Used**:                                      │     │   │
│  │     │    • [Feature 1]: [Weight]                                     │     │   │
│  │     │    • [Feature 2]: [Weight]                                     │     │   │
│  │     │    • [Feature 3]: [Weight]                                     │     │   │
│  │     │                                                             │     │   │
│  │     │ 📊 **Text Analysis**:                                           │     │   │
│  │     │    • Word count: [X]                                            │     │   │
│  │     │    • Unique words: [Y]                                          │     │   │
│  │     │    • Processing time: [Z ms]                                    │     │   │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  🔽 **Advanced Analysis** (Click to expand)                            │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ 📊 **Detailed Probabilities**: [Bar chart of all classes]      │     │   │
│  │     │ 🔍 **Feature Importance**: [Word cloud/feature weights]        │     │   │
│  │     │ 📈 **Model Confidence**: [Confidence distribution]              │     │   │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  │  📁 **Export & Save**                                                  │
│  │     ┌─────────────────────────────────────────────────────────────┐     │   │
│  │     │ [💾 Save Classification] [📊 Export Results] [📋 Add to Batch] │     │
│  │     └─────────────────────────────────────────────────────────────┘     │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  [◀ Previous] [🏠 Start Over] [💾 Save Results]                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 **SIDEBAR - Progress Tracker & Quick Actions**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 📋 PROGRESS TRACKER                                                            │
│                                                                                 │
│ 🎯 Current Step: [3/6] - Column Selection                                     │
│                                                                                 │
│ 📍 Step Status:                                                                │
│   ✅ Step 1: Dataset Selection (Completed)                                    │
│   ✅ Step 2: Data Preprocessing (Completed)                                   │
│   🔄 Step 3: Column Selection (Current)                                       │
│   ⏳ Step 4: Model Configuration (Pending)                                    │
│   ⏳ Step 5: Training Execution (Pending)                                     │
│   ⏳ Step 6: Results Analysis (Pending)                                       │
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
│  📍 STEP [3/6]: Column Selection                                              │
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
- Header: #1f77b4 (Blue)
- Step Indicator: #ff7f0e (Orange)
- Success: #2ca02c (Green)
- Info: #17a2b8 (Info Blue)
- Warning: #ffc107 (Warning Yellow)
- Error: #dc3545 (Error Red)

Wizard Styling:
- Step Header: #f8f9fa (Light Gray)
- Active Step: #e3f2fd (Light Blue)
- Completed Step: #e8f5e8 (Light Green)
- Pending Step: #fff3e0 (Light Orange)

Typography:
- Step Title: 1.5rem, Bold, #1f77b4
- Step Description: 1rem, Normal, #6c757d
- Form Labels: 1rem, Semi-bold, #495057
- Help Text: 0.875rem, Light, #6c757d
```

---

## 🔄 **USER FLOW - Wizard Navigation**

```
1. User lands on Step 1 (Dataset Selection)
   ↓
2. Completes Step 1 → Auto-advance to Step 2
   ↓
3. Completes Step 2 → Auto-advance to Step 3
   ↓
4. Completes Step 3 → Auto-advance to Step 4
   ↓
5. Completes Step 4 → Auto-advance to Step 5
   ↓
6. Completes Step 5 → Auto-advance to Step 6
   ↓
7. Completes Step 6 → Results & Export
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

## 🎯 **KEY FEATURES - Wizard UI**

```
✅ Step-by-Step Workflow
✅ Progressive Disclosure
✅ Auto-advance on Completion
✅ Progress Tracking
✅ Validation at Each Step
✅ Skip to End Option
✅ Mobile Responsive
✅ Session State Management
✅ Error Handling & Recovery
✅ Configuration Persistence
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
        return 'preprocessing_complete' in st.session_state
    elif step == 3:
        return 'columns_selected' in st.session_state
    elif step == 4:
        return 'model_config' in st.session_state
    elif step == 5:
        return 'training_complete' in st.session_state
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

*Giao diện Wizard UI này được thiết kế để tương thích hoàn toàn với Streamlit, dễ lập trình và cung cấp trải nghiệm người dùng tốt nhất theo nguyên tắc Progressive Disclosure.*

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
│  📊 **Confusion Matrix & Performance Metrics**                                │
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
