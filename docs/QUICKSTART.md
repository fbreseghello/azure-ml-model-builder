# Guia de Início Rápido - Azure ML Model Builder

Este guia vai te ajudar a começar rapidamente com o Azure ML Model Builder.

## Pré-requisitos

- Python 3.8 ou superior
- Conta Azure ativa
- Azure ML Workspace configurado

## Instalação em 5 Minutos

### 1. Clone o Repositório

```bash
git clone https://github.com/fbreseghello/azure-ml-model-builder.git
cd azure-ml-model-builder
```

### 2. Crie um Ambiente Virtual

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 4. Configure o Azure ML Workspace

Crie um arquivo `config.json` na raiz do projeto:

```json
{
    "subscription_id": "seu-subscription-id",
    "resource_group": "seu-resource-group",
    "workspace_name": "seu-workspace-name"
}
```

## Primeiro Modelo em 10 Minutos

### Exemplo: Predição de Demanda de Bicicletas

#### 1. Prepare seus Dados

```python
import pandas as pd
from azureml.core import Workspace, Dataset

# Conectar ao workspace
ws = Workspace.from_config()

# Carregar dados de exemplo (CSV)
datastore = ws.get_default_datastore()
dataset = Dataset.Tabular.from_delimited_files(
    path=(datastore, 'bike-rental-data.csv')
)

# Registrar dataset
dataset = dataset.register(
    workspace=ws,
    name='bike-rental-dataset',
    description='Bike rental demand data'
)
```

#### 2. Treine o Modelo

```bash
python scripts/train_model.py \
    --experiment-name bike-rental-quickstart \
    --task regression \
    --dataset-id <seu-dataset-id> \
    --label-column rentals \
    --compute-target cpu-cluster
```

#### 3. Deploy o Modelo

```bash
python scripts/deploy_model.py \
    --model-name bike-rental-model \
    --service-name bike-rental-service \
    --deployment-target aci
```

#### 4. Teste o Modelo

```bash
python scripts/test_model.py \
    --service-name bike-rental-service \
    --use-sample
```

## Uso Básico do AutoML Driver

```python
from automl_driver_refactored import AutoMLDriver

# Configuração
automl_settings = {
    'task_type': 'regression',
    'primary_metric': 'normalized_root_mean_squared_error',
    'experiment_timeout_minutes': 30,
    'enable_early_stopping': True,
    'enable_ensembling': True,
}

# Criar e executar driver
driver = AutoMLDriver(
    automl_settings=automl_settings,
    run_id='experiment-001'
)

result = driver.run(training_percent=100)
print(f"Training completed: {result}")
```

## Estrutura do Projeto

```
azure-ml-model-builder/
├── automl_driver.py              # Driver original (legado)
├── automl_driver_refactored.py   # Driver refatorado
├── scripts/                       # Scripts utilitários
│   ├── train_model.py            # Treinar modelos
│   ├── deploy_model.py           # Deploy de modelos
│   └── test_model.py             # Testar endpoints
├── outputs/                       # Artefatos gerados
├── explanation/                   # Explicações do modelo
└── docs/                          # Documentação
```

## Exemplos Rápidos

### Treinamento com Configuração Customizada

```python
from azureml.train.automl import AutoMLConfig
from azureml.core import Workspace, Dataset, Experiment

ws = Workspace.from_config()
dataset = Dataset.get_by_id(ws, id='dataset-id')

automl_config = AutoMLConfig(
    task='classification',
    training_data=dataset,
    label_column_name='target',
    primary_metric='AUC_weighted',
    n_cross_validations=5,
    enable_early_stopping=True,
    experiment_timeout_minutes=30,
    max_concurrent_iterations=4
)

experiment = Experiment(ws, 'quick-classification')
run = experiment.submit(automl_config)
run.wait_for_completion(show_output=True)
```

### Fazer Predições

```python
import joblib
import pandas as pd

# Carregar modelo
model = joblib.load('outputs/model.pkl')

# Dados de teste
test_data = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6],
    # ... outras features
})

# Predição
predictions = model.predict(test_data)
print(predictions)
```

### Consumir API do Modelo Deployado

```python
import requests
import json

# URL do endpoint
scoring_uri = 'http://your-service.azurecontainer.io/score'

# Dados de entrada
data = {
    'data': [{
        'day': 1,
        'mnth': 1,
        'year': 1,
        'season': 1,
        'holiday': 0,
        'weekday': 1,
        'workingday': 1,
        'weathersit': 1,
        'temp': 0.3,
        'atemp': 0.3,
        'hum': 0.6,
        'windspeed': 0.15
    }]
}

# Fazer request
headers = {'Content-Type': 'application/json'}
response = requests.post(scoring_uri, json=data, headers=headers)

# Resultado
print(response.json())
```

## Próximos Passos

1. **Explore a Documentação Técnica:** [docs/TECHNICAL.md](TECHNICAL.md)
2. **Customize o AutoML:** Ajuste hiperparâmetros e algoritmos
3. **Implemente CI/CD:** Configure pipelines automatizados
4. **Monitore Modelos:** Configure data drift detection
5. **Escale para Produção:** Deploy em AKS com autoscaling

## Recursos Úteis

- [README Principal](../README.md)
- [Documentação Técnica](TECHNICAL.md)
- [Guia de Contribuição](../CONTRIBUTING.md)
- [Azure ML Docs](https://docs.microsoft.com/azure/machine-learning/)

## Troubleshooting Rápido

### Erro de Autenticação

```python
# Use autenticação interativa
from azureml.core.authentication import InteractiveLoginAuthentication
auth = InteractiveLoginAuthentication()
ws = Workspace.from_config(auth=auth)
```

### Timeout no Treinamento

```python
# Reduza o tempo de timeout
automl_settings['experiment_timeout_minutes'] = 15
automl_settings['iteration_timeout_minutes'] = 5
```

### Erro de Memória

```python
# Use amostragem de dados
dataset = dataset.take_sample(probability=0.1)
```

## Suporte

- **Issues:** [GitHub Issues](https://github.com/fbreseghello/azure-ml-model-builder/issues)
- **Discussões:** [GitHub Discussions](https://github.com/fbreseghello/azure-ml-model-builder/discussions)
- **Email:** Abra uma issue no GitHub

---

**Bom treinamento! 🚀**
