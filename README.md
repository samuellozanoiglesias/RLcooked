# Cooked: Multi-Agent Reinforcement Learning Kitchen Simulation

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)
[![Ray RLlib](https://img.shields.io/badge/Ray-2.46.0-orange.svg)](https://ray.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Cooked** is a sophisticated multi-agent reinforcement learning research platform designed for studying cooperative and competitive behaviors in kitchen coordination tasks. Built on a custom game engine, it provides a rich environment where AI agents learn to collaborate or compete while preparing and delivering meals.

## 🎯 Overview

The Cooked platform simulates a kitchen environment where agents must coordinate to:
- 🥬 Gather ingredients from dispensers
- 🔪 Cut vegetables on cutting boards  
- 🥗 Prepare salads by combining ingredients
- 🍽️ Place completed dishes on plates
- 🚚 Deliver finished meals to customers

The platform supports both **cooperative** scenarios (agents work together toward shared goals) and **competitive** scenarios (agents compete for resources and deliveries).

## 🏗️ Architecture

### Core Components

```
cooked/
├── engine/                          # Custom game engine
│   ├── base_game.py                # Base game framework
│   ├── core.py                     # Core engine functionality
│   ├── extensions/                 # Engine extensions and modules
│   └── app/                        # Application runners
├── spoiled_broth/                  # Main game implementation
│   ├── game.py                     # Game logic and mechanics
│   ├── rl/                         # Reinforcement learning components
│   ├── agent/                      # Agent implementations
│   ├── world/                      # Game world and tile system
│   ├── maps/                       # Kitchen layout definitions
│   ├── analysis/                   # Analysis utilities (NEW)
│   └── simulations/                # Simulation utilities (NEW)
├── analysis_*.py                   # Analysis scripts (refactored)
├── experimental_simulation*.py     # Simulation runners (refactored)
└── training-DTDE-spoiled_broth.py # Training orchestration
```

### 🎮 Game Engine

The **Cooked Engine** is a flexible, tile-based game framework supporting:

- **Multi-agent coordination**: Simultaneous agent actions with collision detection
- **Real-time rendering**: 2D sprite-based visualization with web interface
- **Flexible tick rates**: Configurable engine and AI decision frequencies
- **Extensible architecture**: Modular design for easy customization

### 🤖 Reinforcement Learning Framework

Built on **Ray RLlib** with support for:

- **Multi-agent training**: Decentralized (DTDE) and Centralized (CTDE) approaches
- **Advanced algorithms**: PPO, A2C with LSTM support for partial observability
- **Flexible rewards**: Version-controlled reward systems for different scenarios
- **Distributed training**: GPU-accelerated training on computing clusters

## 🚀 Quick Start

### Installation

#### Option 1: Conda Environment (Recommended)
```bash
# Clone the repository
git clone https://github.com/younesStrittmatter/cooked.git
cd cooked

# Create and activate conda environment
conda env create -f environment.yml
conda activate RLcooked

# Install the package in development mode
pip install -e .
```

#### Option 2: Pip Installation
```bash
# Clone the repository
git clone https://github.com/younesStrittmatter/cooked.git
cd cooked

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Training

```bash
# Train agents on a simple kitchen map
python training-DTDE-spoiled_broth.py input_configs/default.txt simple_kitchen_circular 0.001 1 classic v3.1 42

# Parameters:
# - input_configs/default.txt: Configuration file
# - simple_kitchen_circular: Map name
# - 0.001: Learning rate
# - 1: Cooperative mode (0 for competitive)
# - classic: Game version
# - v3.1: Intent/reward version
# - 42: Random seed
```

### Run Simulation

```bash
# Run simulation with trained agents
python experimental_simulation_new.py simple_kitchen_circular 2 v3.1 1 classic training_001 50 \
    --enable-video true --cluster local --duration 180
```

### Analysis

```bash
# Analyze training results
python analysis_classic_new.py v3.1 simple_kitchen_circular --cluster local --smoothing-factor 15

# For competition scenarios
python analysis_competition_new.py v3.1 simple_kitchen_competition --cluster local
```

## 📋 Requirements

### System Requirements

- **Python**: 3.13+
- **GPU**: CUDA-compatible GPU (recommended for training)
- **Memory**: 8GB+ RAM (16GB+ recommended for large-scale training)
- **Storage**: 5GB+ free space

### Key Dependencies

#### Core Framework
- **Ray[rllib]** (2.46.0): Distributed RL training
- **PyTorch** (2.7.0): Deep learning backend
- **Stable-Baselines3** (2.6.0): RL algorithms
- **Gymnasium** (1.0.0): RL environment interface
- **PettingZoo** (1.24.3): Multi-agent RL interface

#### Scientific Computing
- **NumPy** (2.2.5): Numerical computations
- **Pandas** (2.2.3): Data analysis
- **Matplotlib** (3.10.1): Plotting and visualization
- **SciPy** (1.15.3): Scientific computing

#### Web & Visualization
- **Flask** (3.1.0): Web interface backend
- **OpenCV** (4.12.0.88): Video recording and image processing
- **Pillow** (11.2.1): Image manipulation

#### Development & Deployment
- **Jupyter**: Interactive development
- **TensorBoard**: Training monitoring
- **Gunicorn**: Production web server

See `requirements.txt` and `environment.yml` for complete dependency lists.

## 🎯 Features

### 🎮 Game Mechanics

- **Multi-layered cooking process**: Ingredient gathering → Preparation → Assembly → Delivery
- **Dynamic kitchen layouts**: 15+ predefined maps with varying complexity
- **Realistic constraints**: Limited carrying capacity, tool availability, spatial navigation
- **Scoring system**: Points for successful deliveries, penalties for inefficient actions

### 🤖 AI Capabilities

- **Intention-based planning**: Agents form and execute high-level intentions
- **Partial observability**: Realistic information constraints
- **Adaptive behavior**: Agents learn optimal coordination strategies
- **Multi-agent learning**: Support for 1-4 agents per environment

### 📊 Analysis & Monitoring

- **Comprehensive metrics**: Performance, efficiency, coordination measures
- **Real-time visualization**: Live training progress and agent behavior
- **Detailed logging**: Action-level tracking for behavioral analysis
- **Video recording**: Generate MP4 videos of agent interactions

## 🗺️ Maps & Scenarios

### Kitchen Layouts

- **simple_kitchen**: Basic linear layout for learning fundamentals
- **simple_kitchen_circular**: Circular design encouraging coordination
- **simple_kitchen_competition**: Competitive layout with resource scarcity
- **forced_cooperation_kitchen**: Requires mandatory agent collaboration
- **division_of_labor**: Specialized roles for different agents

### Game Modes

#### 🤝 Cooperative Mode
- Shared objectives and rewards
- Agents work together to maximize total deliveries
- Emphasis on communication and coordination

#### ⚔️ Competitive Mode
- Individual agent rewards
- Resource competition
- Strategic behavior emergence

## 🔬 Research Applications

### Supported Research Areas

- **Multi-agent coordination**: Studying emergent cooperation strategies
- **Intent recognition**: Understanding agent planning and communication
- **Curriculum learning**: Progressive difficulty scaling
- **Transfer learning**: Cross-environment agent adaptation
- **Human-AI interaction**: Mixed human-agent teams

### Experimental Configurations

- **Reward shaping**: Customizable reward functions across versions (v1-v3.2)
- **Observation spaces**: Full/partial observability modes
- **Action spaces**: Discrete tile-based interactions
- **Training algorithms**: PPO, A2C, custom policy networks

## 📈 Analysis Tools

### Automated Analysis Pipeline

The refactored analysis system provides:

```bash
# Classic single-agent analysis
python analysis_classic_new.py v3.1 map_name --cluster local

# Competition multi-agent analysis  
python analysis_competition_new.py v3.1 map_name --cluster local

# Pretrained model evaluation
python analysis_pretrained_new.py v3.1 map_name --cluster local
```

### Generated Visualizations

- **Performance curves**: Learning progress over time
- **Action distributions**: Behavioral pattern analysis
- **Coordination metrics**: Multi-agent interaction measures
- **Comparative analysis**: Cross-condition performance evaluation

### Data Outputs

- **Training logs**: Episode-by-episode performance data
- **Action traces**: Detailed agent decision sequences
- **Configuration records**: Complete experimental metadata
- **Statistical summaries**: Aggregated performance metrics

## 🎬 Simulation System

### Professional Simulation Runner

```bash
python experimental_simulation_new.py <map> <agents> <version> <coop> <game_ver> <training_id> <checkpoint>
```

Features:
- **Organized output**: Each simulation in `simulation_{timestamp}/` directory
- **Complete logging**: State, actions, and configuration tracking
- **Video recording**: High-quality MP4 output with HUD overlays
- **Robust execution**: Timeout handling and error recovery

### Output Structure

```
simulations/simulation_2025_10_01-15_30_45/
├── config.txt                           # Complete simulation parameters
├── simulation_log_checkpoint_50_*.csv   # Agent state tracking
├── actions_checkpoint_50_*.csv          # Detailed action logging
└── offline_recording_*.mp4              # Video recording
```

## 🖥️ Development Environment

### Cluster Support

The platform supports multiple computing environments:

#### Local Development
```bash
python script.py --cluster local
```

#### University Clusters
```bash
# Brigit cluster
python script.py --cluster brigit

# Cuenca cluster  
python script.py --cluster cuenca
```

### Development Scripts

Located in `dev-scripts/`:
- `build-requirements.sh`: Dependency management
- `deploy.sh`: Deployment automation
- `config.env`: Environment configuration

## 🧪 Testing

### Unit Tests
```bash
# Test core components
python test_delivery_tracking.py

# Test simulation system
python test_simulation_system.py
```

### Integration Testing
```bash
# Verify full pipeline
python -c "from spoiled_broth.game import SpoiledBroth; print('✓ Game engine working')"
python -c "from spoiled_broth.rl.game_env import SpoiledBrothEnv; print('✓ RL environment working')"
```

## 📚 Documentation

### API Documentation

Key classes and methods:

```python
# Game creation
from spoiled_broth.game import SpoiledBroth
game = SpoiledBroth(map_nr=\"simple_kitchen\", grid_size=(8,8), intent_version=\"v3.1\")

# RL environment
from spoiled_broth.rl.game_env import SpoiledBrothEnv  
env = SpoiledBrothEnv(map_name=\"simple_kitchen\", intent_version=\"v3.1\")

# Analysis utilities
from spoiled_broth.analysis.utils import main_analysis_pipeline
results = main_analysis_pipeline(\"classic\", \"v3.1\", \"simple_kitchen\")

# Simulation utilities
from spoiled_broth.simulations.utils import main_simulation_pipeline
outputs = main_simulation_pipeline(\"simple_kitchen\", 2, \"v3.1\", True, \"classic\", \"training_001\", 50)
```

### Configuration

Key configuration files:
- `spoiled_broth/config.py`: Core game parameters
- `environment.yml`: Conda environment specification
- `requirements.txt`: Python package dependencies

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes following the established code style
4. Add tests for new functionality
5. Run the test suite: `python -m pytest`
6. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Document classes and methods with docstrings
- Maintain the modular architecture

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ray Team**: For the excellent RLlib framework
- **OpenAI**: For Gymnasium standards
- **Research Community**: For multi-agent RL foundations
- **Contributors**: All developers who have contributed to this project

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/younesStrittmatter/cooked/issues)
- **Discussions**: [GitHub Discussions](https://github.com/younesStrittmatter/cooked/discussions)
- **Email**: Contact the development team (samuel.lozano@ucm.es or ys5852@princeton.edu)

## 🗓️ Roadmap

### Upcoming Features

- **Web-based training interface**: Real-time monitoring dashboard
- **Advanced curriculum learning**: Automated difficulty progression
- **Human-in-the-loop training**: Mixed human-agent scenarios
- **Extended environments**: New kitchen layouts and challenges
- **Model interpretability**: Agent decision explanation tools

---

**Cooked** - *Where AI agents learn to cook together* 🍳🤖