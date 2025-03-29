# Yamnet Docker Analysis

A Python application for analyzing audio files using YAMNet and Llama models, with Docker support.

## Features

- Audio file analysis using YAMNet
- Crowd reaction detection and analysis
- Llama-based insights generation
- Docker containerization
- RESTful API (coming soon)

## Prerequisites

- Python 3.8+
- Docker
- Docker Compose
- Required model files:
  - `models/llama-2-7b-chat.gguf`
  - `models/yamnet/yamnet_model/`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Omersh30/Yamnet_Docker.git
cd Yamnet_Docker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up required model files in the `models/` directory.

## Usage

### Local Development

Run the application locally:
```bash
python src/main.py
```

### Docker

Build and run using Docker:
```bash
docker-compose up --build
```

## Project Structure

```
Yamnet_Docker/
├── src/
│   ├── api/           # API endpoints (coming soon)
│   ├── audio_processing/
│   ├── llama_analysis/
│   └── main.py
├── tests/             # Test files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Development

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Description of your changes"
```

3. Push to your branch:
```bash
git push origin feature/your-feature-name
```

4. Create a Pull Request

## License

MIT License 