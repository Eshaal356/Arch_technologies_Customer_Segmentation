# Arch Prism Intelligence Core | Customer Segmentation Platform

## Overview

**Arch Prism** is a sophisticated, high-performance intelligence platform built with Streamlit that redefines customer segmentation by treating **Nations as Customers**. By analyzing global health, economic, and social data, the platform identifies distinct behavioral cohorts to drive strategic decision-making.

This application leverages advanced machine learning techniques—including **K-Means Clustering**, **Principal Component Analysis (PCA)**, and **Multi-Layer Perceptron (MLP) Regressors**—to transform raw data into actionable insights, visualized through a stunning, "Neural/Cyber" aesthetic interface.

## 🚀 Distinctive Features

*   **Multi-Dimensional Segmentation**: Classifies nations into 4 strategic tiers using an advanced multi-pillar framework (Demographic, Psychographic, Behavioral, Geographic).
*   **Geospatial Intelligence**: Features an interactive **3D Orthographic Globe** (`Geospatial Intelligence` page) to visualize the spatial distribution of segments across the planet.
*   **3D Neural Manifold**: A 3D PCA visualization that projects 22-dimensional feature space into a navigable 3D cluster map.
*   **Nation Benchmarking Radar**: "Nation Duels" feature allowing side-by-side comparison of countries across normalized health and economic vectors.
*   **Predictive Analytics**: Integrated MLP Neural Network to potential life expectancy outcomes based on socio-economic inputs.
*   **Strategic Roadmap**: A visual representation of the data science pipeline, execution maturity, and risk topology.
*   **High-End UI/UX**: Custom "Arch Prism" design system featuring glassmorphism, neon accents (`#a855f7`, `#22d3ee`), and fluid animations.

## 📂 Project Structure

*   `streamlit_app.py`: The main entry point for the application containing all logic for the UI, data processing, and machine learning models.
*   `Life Expectancy Data.csv`: The core dataset containing global health and economic indicators (WHO data).
*   `arch_technologies_customer_segmentation (2).py`: Comparative analysis script.
*   `neural_architecture.png`: Asset used for visual identity.

## 🛠️ Installation & Usage

### Prerequisites

Ensure you have Python installed. You will need the following libraries:

*   streamlit
*   pandas
*   numpy
*   plotly
*   scikit-learn

### Setup

1.  **Clone the repository** (if applicable) or download the project files.
2.  **Install dependencies**:
    ```bash
    pip install streamlit pandas numpy plotly scikit-learn
    ```
3.  **Run the Application**:
    ```bash
    streamlit run streamlit_app.py
    ```

## 🧠 Intelligence Engine

The platform operates on a "Data Utility Engine" that processes the `Life Expectancy Data.csv`. Key steps include:
1.  **Data Ingestion & Cleaning**: Handling missing values and normalizing features using `RobustScaler`.
2.  **Dimensionality Reduction**: Reducing 22 features to 3 principal components using PCA for visualization and noise reduction.
3.  **Clustering**: Applying K-Means to identify 4 distinct "Customer" Segments (Tier 1 - Tier 4).
4.  **Neural Modeling**: Training an MLP Regressor to predict Life Expectancy based on key drivers like Alcohol consumption, GDP, and Schooling.

## 🎨 Design Philosophy

The interface uses a custom theme focusing on "Dark Mode" aesthetics with:
-   **Primary Color**: Purple (`#a855f7`)
-   **Secondary Color**: Cyan (`#22d3ee`)
-   **Background**: Deep Space Blue/Black gradients (`#020617` to `#0f172a`)
-   **Fonts**: 'Plus Jakarta Sans', 'Outfit', and 'Inter' for a modern, tech-forward look.

---
