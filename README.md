# ETH Vegas Tunnel Python

## Description
This project interacts with the Gate.io API to fetch cryptocurrency futures data, calculate technical indicators, and make trading decisions.

## Setup Instructions
1. Clone the repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Gate.io API key and secret as environment variables:
   ```bash
   export API_KEY=your_api_key
   export API_SECRET=your_api_secret
   ```

## Usage
Run the main script to fetch market data and calculate trading signals:
```bash
python ETH_Vegas_Tunnel.py
```

## Dependencies
- Python 3.8+
- pandas
- gate-api

## License
This project is licensed under the MIT License.