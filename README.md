# Azure ML AutoML Model Builder

> Sistema automatizado de construção e deploy de modelos de Machine Learning usando Azure AutoML

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Azure ML](https://img.shields.io/badge/Azure_ML-Supported-0078D4.svg)](https://azure.microsoft.com/services/machine-learning/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Sobre o Projeto

Este projeto implementa um pipeline completo de AutoML (Automated Machine Learning) usando Azure Machine Learning. Ele automatiza o processo de treinamento, avaliação e deploy de modelos de machine learning, incluindo:

- Treinamento automatizado com Azure AutoML
- Geração de explicabilidade do modelo (Model Interpretability)
- Scripts de scoring otimizados para produção
- Integração com MLflow para rastreamento
- Suporte a diferentes tipos de tarefas (Classificação, Regressão)

## Características

- **AutoML Completo**: Treinamento automatizado com seleção de algoritmos e hiperparâmetros
- **Interpretabilidade**: Análise de importância de features e explicações do modelo
- **MLflow Integration**: Rastreamento de experimentos e versionamento de modelos
- **Production Ready**: Scripts de scoring otimizados para deployment
- **Validação Cruzada**: Suporte a diferentes estratégias de validação
- **Data Prep**: Pipeline de preparação de dados integrado
- **Scoring Flexível**: Múltiplas versões de scripts de scoring (v1, v2, Power BI)

## Arquitetura

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Data Source   │─────▶│  AutoML Driver   │─────▶│  Trained Model  │
│  (Azure Store)  │      │  (Training)      │      │   (MLflow)      │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                  │                          │
                                  ▼                          ▼
                         ┌──────────────────┐      ┌─────────────────┐
                         │  Explainability  │      │ Scoring Scripts │
                         │   (Interpret)    │      │  (Deployment)   │
                         └──────────────────┘      └─────────────────┘
```

## Pré-requisitos

- Python 3.8+
- Conta Azure com Azure Machine Learning workspace
- Azure CLI (opcional, para deployment)
- Conda ou venv para ambiente virtual

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/fbreseghello/azure-ml-model-builder.git
cd azure-ml-model-builder
```

### 2. Crie um ambiente virtual

```bash
# Usando conda
conda create -n azure-ml python=3.8
conda activate azure-ml

# Ou usando venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

## Configuração

### Azure ML Workspace

1. Crie um workspace no Azure Machine Learning
2. Configure as credenciais de acesso:

```python
from azureml.core import Workspace

ws = Workspace(
    subscription_id='<your-subscription-id>',
    resource_group='<your-resource-group>',
    workspace_name='<your-workspace-name>'
)
```

### Configuração do Dataset

Configure seu dataset no Azure ML ou use o Data Prep pipeline incluído no projeto.

## Uso

### Treinamento do Modelo

```bash
python automl_driver.py
```

O script irá:
1. Conectar ao Azure ML workspace
2. Carregar e preparar os dados
3. Executar o AutoML training
4. Gerar explicações do modelo
5. Salvar artefatos em `outputs/`

### Usando o Modelo Treinado

```python
import joblib
import pandas as pd

# Carregar o modelo
model = joblib.load('outputs/mlflow-model/model.pkl')

# Fazer predições
data = pd.DataFrame({
    'day': [1],
    'mnth': [1],
    'year': [0],
    # ... outras features
})

predictions = model.predict(data)
```

### Deploy para Azure

```bash
# Usando o script de scoring gerado
python outputs/scoring_file_v_2_0_0.py
```

## Estrutura do Projeto

```
azure-ml-model-builder/
├── automl_driver.py              # Script principal de treinamento
├── requirements.txt               # Dependências do projeto
├── README.md                      # Documentação
├── .gitignore                     # Arquivos ignorados pelo git
│
├── outputs/                       # Artefatos gerados
│   ├── mlflow-model/             # Modelo MLflow
│   ├── scoring_file_v_2_0_0.py   # Script de scoring (produção)
│   ├── conda_env_v_1_0_0.yml     # Ambiente conda
│   ├── pipeline_graph.json        # Visualização do pipeline
│   └── generated_code/            # Código gerado automaticamente
│       ├── script.py              # Script de treino reproduzível
│       └── script_run_notebook.ipynb
│
└── explanation/                   # Explicabilidade do modelo
    ├── 2cbd3df5/                 # Explicações globais
    └── 66c77c2a/                 # Explicações locais
```

## Modelo Treinado

O projeto inclui um modelo treinado para predição de demanda de bicicletas compartilhadas, com as seguintes características:

### Features
- **Temporais**: day, month, year, season, weekday
- **Contextuais**: holiday, workingday, weathersit
- **Ambientais**: temp, atemp, hum, windspeed

### Métricas de Performance
Os resultados do treinamento estão disponíveis em:
- `outputs/pipeline_graph.json` - Estrutura do pipeline
- `explanation/` - Análise de importância de features

## Deploy

### Azure Container Instance (ACI)

```python
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

ws = Workspace.from_config()
model = Model(ws, 'seu-modelo')

inference_config = InferenceConfig(
    entry_script='outputs/scoring_file_v_2_0_0.py',
    environment=env
)

aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1
)

service = Model.deploy(
    workspace=ws,
    name='bike-sharing-service',
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
print(f"Service URL: {service.scoring_uri}")
```

### Power BI Integration

Use `outputs/scoring_file_pbi_v_1_0_0.py` para integração direta com Power BI.

## Explicabilidade do Modelo

O projeto gera automaticamente explicações do modelo usando Azure ML Interpret:

```python
from azureml.interpret import ExplanationClient

# As explicações são salvas automaticamente em explanation/
# - global_importance: importância global de features
# - local_importance: explicações por predição
# - visualization_dict: dados para dashboards
```

## Desenvolvimento

### Executar Testes

```bash
pytest tests/
```

### Formatação de Código

```bash
black .
flake8 .
mypy .
```

**Nota**: Este projeto foi atualizado em janeiro de 2026 para incluir as melhores práticas mais recentes de Azure ML e Python.
