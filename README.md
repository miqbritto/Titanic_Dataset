# Titanic - Previsão de Sobreviventes com Machine Learning

Este repositório contém um projeto de Machine Learning desenvolvido para prever a sobrevivência de passageiros do Titanic com base em dados tabulares. Todo o processamento de dados e modelagem foi feito com as etapas organizadas manualmente.

---

## 📚 Descrição do Projeto

O objetivo é aplicar técnicas de pré-processamento, engenharia de atributos e modelagem para prever quem sobreviveu ao naufrágio do Titanic. O modelo final é treinado com `RandomForestClassifier` e gera um arquivo `submission.csv` com as previsões para submissão no Kaggle.

---

## 🛠️ Tecnologias Utilizadas

- Python 3
- pandas
- numpy
- scikit-learn

---

## 📁 Etapas do Processo

### 1. Leitura dos dados
Os arquivos `train.csv` e `test.csv` são carregados a partir do ambiente do Kaggle.

### 2. Tratamento de valores ausentes
- `Fare` e `Age`: preenchidos com média.
- `Embarked`: preenchido com a moda (valor mais frequente).
- `CabinLetter`: extraído da primeira letra de `Cabin`, e valores ausentes preenchidos com `"Unknown"`.

### 3. Engenharia de atributos (Feature Engineering)
Foram criadas novas variáveis para enriquecer o modelo:
- `CabinLetter`: letra inicial da cabine
- `FareBin`: discretização da tarifa em 10 faixas (`qcut`)
- `AgeBin`: faixas etárias manuais (criança, jovem, adulto, idoso)
- `FamilySize`: soma de irmãos/cônjuges (`SibSp`) e pais/filhos (`Parch`) + 1 (o próprio passageiro)
- `IsAlone`: se o passageiro estava sozinho
- `FarePerPerson`: tarifa dividida pela quantidade de pessoas na família
- `ClassFareInteraction`: multiplicação entre a classe (`Pclass`) e a tarifa (`Fare`)

### 4. Codificação de variáveis categóricas
Foi utilizado `OneHotEncoder` do scikit-learn para transformar:
- `Sex`
- `Embarked`
- `CabinLetter`
- `IsAlone` (embora binário, foi incluído no mesmo grupo)

  Essas são variáveis categóricas e foram convertidas para variáveis discretas, pois a maiorias dos algoritmos de ML não lida bem com variáveis categóricas.

### 5. Modelagem
O modelo escolhido foi o `RandomForestClassifier` com os melhores hiperparâmetros definidos através da biblioteca RandomizedSearchCV do scikit-learn:
```python
RandomForestClassifier(
    n_estimators=500, 
    criterion='entropy', 
    max_depth=8, 
    min_samples_split=8, 
    min_samples_leaf=2
)
