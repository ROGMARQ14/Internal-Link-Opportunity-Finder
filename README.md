# Internal Link Opportunity Finder

A Streamlit app to discover internal linking opportunities for your website using vector embeddings.

## About

This tool helps website owners and SEO specialists find contextually relevant internal linking opportunities based on vector embeddings. It analyzes your website's current internal linking structure and identifies related pages that would benefit from being linked together.

The app is based on [Everett Sizemore's tutorial on Moz](https://moz.com/blog/internal-linking-opportunities-with-vector-embeddings), with a user-friendly interface built using Streamlit.

## Features

- Upload and clean internal link data (e.g., from Screaming Frog)
- Process and analyze vector embeddings for your pages
- Identify contextually similar pages based on their content
- Find missing internal linking opportunities between related pages
- Generate downloadable reports in Excel and CSV formats
- Filter results to focus on the most important opportunities

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- openpyxl

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/internal-link-finder.git
cd internal-link-finder

2. Install the required packages:
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run app.py

## Usage

### Step 1: Prepare Your Data

You'll need two CSV files:

1. **Internal Links Export**:
- Export all internal links from your website using a crawler like Screaming Frog
- The file should contain at minimum: Source URL, Destination URL, and Anchor Text

2. **Page Embeddings**:
- Generate vector embeddings for your pages' content
- The file should contain: URL and embeddings data (vector representations of each page)

### Step 2: Upload and Process

1. Upload your internal links CSV file
2. Clean the links data
3. Upload your embeddings CSV file
4. Clean the embeddings data
5. Run the analysis to find internal linking opportunities

### Step 3: Review and Download Results

1. Filter the results to focus on specific opportunities
2. Review the highlighted missing links
3. Download the full report in Excel or CSV format
4. Implement the suggested internal links on your website

## Deployment

### Local Deployment

Run the app locally with:
streamlit run app.py

### Cloud Deployment (Streamlit Cloud)

1. Push your code to GitHub
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app, pointing to your GitHub repository
4. Deploy and share the app URL

## Tips for Best Results

- Use a comprehensive crawl of your website to capture all existing internal links
- Generate embeddings that accurately represent the content of each page
- Focus on implementing the most valuable linking opportunities first
- Re-run the analysis periodically to find new opportunities as your site grows

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Everett Sizemore for the original methodology
- Britney Muller for the initial script
