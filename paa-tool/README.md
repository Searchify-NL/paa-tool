# SEO SERP Real-Estate Visibility Explorer

A Streamlit-based tool for measuring and exploring SERP real-estate visibility by analyzing keyword data from Surplex KWR and SEMrush exports, plus live Google SERP fetching via SearchAPI.io.

## Features

### 📊 Data Import & Processing
- **Surplex KWR format**: Handles `Surplex DE KWR 6-25.xlsx` files with "Unnamed" columns
- **SEMrush export**: Processes `rituals.com-organic.Positions-*.xlsx` files with "Previous position" column
- **Auto-detection**: Automatically identifies file format and normalizes data
- **Multi-file upload**: Combine multiple Excel exports into a single dataset

### 🔍 Live SERP Analysis
- **SearchAPI.io integration**: Fetch live Google SERPs for any keywords
- **Feature extraction**: Parse organic results, PAA questions, featured snippets, videos, images, etc.
- **Real-estate scoring**: Compute visibility scores based on CTR benchmarks and SERP features
- **Domain filtering**: Focus scoring on specific domains

### 📈 Analytics & Visualization
- **CTR benchmarks**: Latest Google Organic CTR data from Advanced Web Ranking
- **SERP scoring**: `(1 + #extra_features) × CTR_benchmark` for uploaded data
- **Live scoring**: Comprehensive scoring for organic, PAA, features, and featured snippets
- **Interactive charts**: Top 10 keywords by score with bar charts

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure SearchAPI.io (Optional)
For live SERP fetching, add your SearchAPI key to Streamlit secrets:

**Option A: Create `.streamlit/secrets.toml`**
```toml
SEARCHAPI_KEY = "your_searchapi_key_here"
```

**Option B: Set environment variable**
```bash
export SEARCHAPI_KEY="your_searchapi_key_here"
```

### 3. Run the App
```bash
streamlit run app.py
```

## Usage

### Tab 1: Upload & Preview
1. **Upload Excel files**: Drag and drop Surplex KWR or SEMrush export files
2. **Preview data**: View normalized and combined dataset
3. **Analyze scores**: See top 10 keywords by SERP real-estate score
4. **Download results**: Export normalized data as CSV

### Tab 2: Live SERP Fetch
1. **Select keywords**: Choose from your uploaded dataset
2. **Configure location**: Set target country (US, UK, DE, FR, ES, IT, CA, AU, NL, BR, MX, JP, KR, CN, IN, SE, NO, DK, FI, PL, CZ, HU, RO, BG, HR, RS, SK, SI, EE, LV, LT, IE, NZ, ZA, SG, MY, PH, TH, VN, ID, TR, GR, PT, BE, CH, AT, LU, MT, CY, etc.)
3. **Optional domain**: Filter organic results to specific domain
4. **Fetch & analyze**: Get live SERP data and compute scores
5. **Append results**: Add live data to your existing dataset

## Data Formats

### Surplex KWR Format
Expected columns:
- `Keywords`
- `Total Search Volume` (under "Unnamed" headers)
- `Position`
- `SERP Features by Keyword`
- `Keyword Intents`
- `Position Type`
- `Timestamp`

### SEMrush Export Format
Expected columns:
- `Keyword`
- `Position`
- `Previous position`
- `Search Volume`
- `URL`
- `Traffic`
- `Traffic (%)`
- `SERP Features by Keyword`
- `Keyword Intents`
- `Position Type`
- `Timestamp`

## Scoring Methodology

### Uploaded Data Scoring
```
SERP Score = (1 + number_of_extra_features) × CTR_benchmark
```

### Live SERP Scoring
```
Total Score = Organic Score + PAA Score + Feature Score + Featured Snippet Bonus

Where:
- Organic Score = Σ(CTR_benchmark for organic results on your domain)
- PAA Score = (avg_CTR_pos_1_2) × number_of_PAA_questions
- Feature Score = Σ(feature_weight × feature_count)
- Featured Snippet Bonus = 0.20 if present
```

### Feature Weights
- **Video**: 0.15 (high engagement)
- **Shopping**: 0.12 (commercial intent)
- **Image**: 0.10 (visual appeal)
- **News**: 0.08 (timeliness)
- **Local**: 0.06 (local intent)

## API Integration

### SearchAPI.io
- **Endpoint**: `https://api.searchapi.io/google`
- **Rate limiting**: 1 second delay between requests
- **Error handling**: Graceful fallback for rate limits and API errors
- **Features extracted**: Organic results, PAA, featured snippets, videos, images, news, shopping, local

### CTR Benchmarks
- **Source**: Advanced Web Ranking (live fetch with fallback)
- **Fallback**: Industry-agnostic curve based on AWR Q1 2025 data
- **Coverage**: Positions 1-10 with decimal CTR values

## File Structure
```
paa-tool/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Error Handling

The app includes comprehensive error handling for:
- **File upload errors**: Invalid Excel files, missing columns
- **API rate limits**: Automatic retry with delays
- **Network timeouts**: Graceful fallback to static data
- **Data parsing errors**: Skip invalid rows, continue processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample data
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 