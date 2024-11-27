from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# web framework for building APIs with Python. 
app = FastAPI()

# For showing background images on web.
app.mount("/static", StaticFiles(directory="C:/Users/User/Desktop/PAI PROJECT/env/static"), name="static")
# Load dataset
df = pd.read_csv('env/WineQT.csv').dropna()
df.drop(labels=['Id'], axis=1, inplace=True)

# Function to generate base64 plot
def generate_base64_plot(plot_type: str, n_clusters: int = 2):
    plt.figure()  # Clear the previous plot
    buffer = BytesIO()  # Buffer to hold the image

    try:
        if plot_type == "correlation":
            # Correlation heatmap
            corr_matrix = df.corr()
            plt.figure(figsize=(10, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
            plt.title('Correlation Matrix')

        elif plot_type == "scatter":
            # Scatterplot example
            plt.figure(figsize=(10, 10))
            sns.scatterplot(x='pH', y='alcohol', data=df)
            plt.title("Scatter plot of pH vs Alcohol")

        elif plot_type == "box_all":
            # Box plot for all numeric features
            plt.figure(figsize=(10, 10))
            sns.boxplot(data=df.select_dtypes(include=['float64']))
            plt.ylim(0,200)
            plt.title("Box Plot of All Features")
            plt.xticks(rotation=30)

        elif plot_type == "box_melted":
            # Bar graph of all numerical plots
            df_melted_box = df.melt(value_vars=df.select_dtypes(include=['float64', 'int64']).columns)
            plt.figure(figsize=(10, 10))
            sns.barplot(x='variable', y='value', data=df_melted_box,ci=None) 
            plt.ylim(0,50)
            plt.title('Bar Graph of All Numerical Features')
            plt.xticks(rotation=45)

        elif plot_type == "distribution":
            # Distribution of all numerical features
            df_melted_hist = df.melt(value_vars=df.select_dtypes(include=['float64', 'int64']).columns)
            plt.figure(figsize=(10, 10))
            sns.histplot(df_melted_hist, x='value', hue='variable', multiple='stack', bins=30, kde=True)
            plt.title('Distribution of All Numerical Features')

        elif plot_type == "kmeans":
            # K-Means Clustering
            numeric_features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['quality', 'Id'], errors='ignore') #Extracts Numeric data and remove quality and id
            scaler = StandardScaler() #Ensures all numeric features are scaled to have a mean of 0 and a standard deviation of 1.
            scaled_data = scaler.fit_transform(numeric_features)

            # Apply PCA for visualization
            pca = PCA(n_components=2) #reduce the high-dimensional data to 2 dimensions
            reduced_data = pca.fit_transform(scaled_data) # Identifies the directions (principal components) that maximize variance in the data.

            # K-Means Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)

            # Create a DataFrame with PCA results and clusters
            pca_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = clusters

            # Plot the clusters
            plt.figure(figsize=(10, 10))
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='tab10', data=pca_df, s=50)
            plt.title(f'K-Means Clustering with {n_clusters} Clusters (PCA)')
            plt.legend(title="Cluster")

        else:
            return None

        plt.savefig(buffer, format="png")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()
        return base64_image

    except Exception as e:
        print(f"Error generating plot: {e}")
        return None

# Route to serve HTML with a specific plot
@app.get("/", response_class=HTMLResponse)
async def home(
    plot_type: str = Query("correlation", description="Type of plot to generate"),
    n_clusters: int = Query(3, description="Number of clusters for K-Means (if applicable)")
):
    base64_image = generate_base64_plot(plot_type, n_clusters)
    if not base64_image:
        return HTMLResponse(content="<h1>Error: Could not generate plot.</h1>", status_code=500)

    with open("env/index.html", "r") as file:
        html_content = file.read()
    return html_content.replace("{{base64_image}}", base64_image)