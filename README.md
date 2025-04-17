# 📊 Crime Incidents in 2024 – Exploratory Data Analysis (EDA)

**Author:** Priyam Tiwari  
**Course Code:** INT375  
**University:** Lovely Professional University  
**Guided by:** Mrs. Aashima, SCSE (Data Science)  
**GitHub Repo:** [crime_incident2024-EDA](https://github.com/1priyam/crime_incident2024-EDA.git)
 
***************

## 📝 Project Overview 

This project provides a deep dive into crime incidents reported in **Washington, D.C. during 2024**, with the aim of uncovering hidden patterns, identifying high-risk areas, and informing public safety strategies. Through **Exploratory Data Analysis (EDA)** techniques, this analysis visualizes crime distribution across time, geography, and categories such as offense types and law enforcement responses.

---
  
## 🎯 Objectives

- Analyze temporal crime trends (monthly, daily, hourly)
- Identify the most frequent types of crime
- Map out geographical hotspots and ward-wise distribution
- Study police response times and clearance status
- Detect outliers and correlate numerical variables
- **NEW:** Explore temporal-geographic patterns (e.g., seasonal hotspots)

---

## 📂 Dataset

- **Source:** [DC Open Data Portal](https://opendata.dc.gov/)
- **Content:** Official reports from the Metropolitan Police Department
- **Attributes:**  
  - Offense types  
  - Date and time of incidents  
  - Method used  
  - Location details (Neighborhoods, Wards, GPS)  
  - Police response and case status  

---

## 🛠️ Tools and Libraries

- Python  
- Pandas  
- Matplotlib  
- Seaborn  
- NumPy  
- SciPy (z-score outlier detection)

---

## 📈 Key Analysis Sections

### 1. Temporal Analysis
- Crimes by **Month**, **Day of the Week**, and **Hour of Day**
- Identifies peak crime times (e.g., late evenings, Fridays)

### 2. Crime Type Distribution
- Top 10 most common crimes (e.g., THEFT/OTHER, THEFT F/AUTO)
- Method of offense analysis for severity insight

### 3. Geographic Analysis
- Crimes by **Neighborhood Cluster**
- Crime distribution by **Ward**
- Pinpoints urban hotspots like Columbia Heights and Shaw

### 4. Law Enforcement Response
- **Response time** analysis (from incident to report)
- **Clearance status** visualization (DISPOSITION)

### 5. Temporal-Geographic Correlation (NEW)
- Seasonal hotspots in high-crime neighborhoods
- Placeholder for event-based analysis (requires event dataset)

### 6. Correlation Matrix
- Heatmap showing correlations between numerical features

### 7. Outlier Detection
- Boxplot visualization  
- Z-score-based numerical outlier count

---

## 📊 Visual Output Samples

The project generates:
- Line plots for temporal patterns
- Bar charts for offense frequency and location-based stats
- Heatmaps for correlation
- Histograms for police response time
- Dark-themed plots for modern aesthetics

---

## 📌 Key Findings

- **Theft-related crimes** dominate incident types.
- **Fridays** and **late afternoons/evenings** are peak times.
- **Neighborhoods like Columbia Heights and Shaw** are major hotspots.
- **Wards 1 and 6** have the highest crime densities.
- Most police responses occur within **100 minutes**.

---

## 🔮 Future Scope

- Integration with **GIS systems** for real-time mapping
- Use of **Machine Learning models** for predictive policing
- Incorporate **event data** and **demographics** for deeper insights
- Build **real-time dashboards** for live monitoring

---

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/1priyam/crime_incident2024-EDA.git
   cd crime_incident2024-EDA
