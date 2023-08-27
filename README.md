# eye-tracking-opencv

Os códigos dentro da pasta `teste_de_conceito` são testes que foram executados para avaliar a viabilidade de diversas formas de implementação das soluções propostas.

A combinação que teve o melhor resultado foi analisar o rosto com o mediapipe e extrair os 468 landmarks dele e, a partir destes landmarks, selecionar a região dos olhos.

Com a imagem correspondente a cada olho foi-se feite uma analise de histograma das cores da imagem, apos alguns filtros serem aplicados, para tentar analisar onde estavam a iris e a pupila. O resultado foi que a pupia é razoavelmente fácil e consistente de se obter, contudo, a iris não. Por conta disso será adicionada mais uma etapa no processo de analise, com um outro algorítimo da mediapipe, o qual extrai a posição da iris de um rosto.

## Pré-requisitos
- Python 3.9
- pipenv
## Como rodar

Eu utilizo o gerenciador de pacotes pipenv, caso não o tenha instalado execute `pip install pipenv`

Para rodar o programa:

- `pipenv shell`
- `pipenv install`
- `python main.py [argumentos]`

## Criar o executavel

```
pipenv run pyinstaller --onefile --paths [seu_caminho] ./main.py --add-data '[seu_caminho]\Lib\site-packages\mediapipe\modules:mediapipe/modules' --add-data '[seu_caminho]\Lib\site-packages\mediapipe\python\solutions:mediapipe/solutions'

```
Substitua [seu_caminho] pelo caminho apropriado do seu ambiente.
## Regras

PROIBIDO colocar pastas que contenham o caractere '.' no nome dentro do diretório /raw para o processamento em lote

## Documentação de Argumentos CLI

### Descrição
O script aceita vários argumentos de linha de comando para configurar suas opções globais. Abaixo estão as opções disponíveis e suas descrições.

### Argumentos

#### -showprocess
Define se o processo será mostrado.

- Valores aceitos: s (sim) ou n (não)
- Valor padrão: n

#### -drawbb
Define se a bounding box será desenhada.

- Valores aceitos: s (sim) ou n (não)
- Valor padrão: n

#### -drawir
Define se a íris será desenhada.

- Valores aceitos: s (sim) ou n (não)
- Valor padrão: n

#### -drawpu
Define se a pupila será desenhada.

- Valores aceitos: s (sim) ou n (não)
- Valor padrão: n

#### -drawpp
Define se as posições passadas serão desenhadas.

- Valores aceitos: s (sim) ou n (não)
- Valor padrão: s

#### -drawmp
Define se os pontos da máscara serão desenhados.

- Valores aceitos: s (sim) ou n (não)
- Valor padrão: n

#### -drawgz
Define se o olhar será desenhado.

- Valores aceitos: s (sim) ou n (não)
- Valor padrão: n

#### -showwarn
Define se os avisos serão mostrados.

- Valores aceitos: s (sim) ou n (não)
- Valor padrão: s

#### -multicore
Define se o processamento multicore será usado.

- Valores aceitos: s (sim) ou n (não)
- Valor padrão: n

#### -overwrite
Define se os arquivos existentes serão sobrescritos. Se -multicore for s, o valor padrão é s. Caso contrário, o valor padrão é n.

- Valores aceitos: s (sim) ou n (não)

#### -path
Define o caminho para os arquivos.

- Valor padrão: ""./vds"
##### Exemplo de Uso

`python main.py -showprocess s -drawbb n -path "./meus_arquivos"`
>Nota: Se um argumento inválido for fornecido para uma opção, o script imprimirá uma mensagem de erro e encerrará sua execução.