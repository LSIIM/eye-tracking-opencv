# eye-tracking-opencv

Os códigos dentro da pasta `teste_de_conceito` são testes que foram executados para avaliar a viabilidade de diversas formas de implementação das soluções propostas.

A combinação que teve o melhor resultado foi analisar o rosto com o mediapipe e extrair os 468 landmarks dele e, a partir destes landmarks, selecionar a região dos olhos.

Com a imagem correspondente a cada olho foi-se feite uma analise de histograma das cores da imagem, apos alguns filtros serem aplicados, para tentar analisar onde estavam a iris e a pupila. O resultado foi que a pupia é razoavelmente fácil e consistente de se obter, contudo, a iris não. Por conta disso será adicionada mais uma etapa no processo de analise, com um outro algorítimo da mediapipe, o qual extrai a posição da iris de um rosto.

## Como rodar

Eu utilizo o gerenciador de pacotes pipenv, caso não o tenha instalado execute `pip install pipenv`

Para rodar o programa:

- `pipenv shell`
- `pipenv install`
- `python eye_tracker.py`
