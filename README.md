# LED Inspector

Aplicação desktop para inspecionar LEDs com câmera contínua, imagem golden, região de interesse e análise por inteligência artificial local.

## Interface principal

A interface usa as duas bibliotecas solicitadas:

- **CustomTkinter** para janelas, botões, campos e aparência moderna;
- **Tkinter Canvas** para exibir as imagens e desenhar a região de interesse com o mouse;
- **OpenCV** para manter a câmera ativa continuamente;
- **Ollama** para comparar a golden com a peça atual usando um modelo de visão.

## Como abrir no Windows

Dê dois cliques em `run_app.bat`. Na primeira execução, o arquivo instala as dependências e abre a interface desktop.

Também é possível iniciar pelo terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python desktop_app.py
```

## Fluxo de inspeção

1. Ao abrir, o programa procura automaticamente uma câmera nos índices `0` a `4`.
2. Se necessário, informe outro índice ou mantenha `auto` e clique em **Iniciar câmera**. A imagem ficará ativa durante todo o processo.
3. Se nenhuma câmera for encontrada, conecte a câmera, feche outros programas que possam estar usando-a e confira em **Configurações do Windows > Privacidade e segurança > Câmera** se o acesso está permitido.
4. Coloque uma peça comprovadamente boa no suporte.
5. Clique em **Capturar golden**.
6. Na imagem golden, clique e arraste sobre a placa para marcar a região de interesse.
7. Para inspecionar somente uma LED, mantenha **Região marcada = uma única LED** ativado e marque apenas essa peça. Para identificar individualmente várias LEDs dentro da seleção, desative essa opção e informe as linhas e colunas corretas (por exemplo, `1` linha e `2` colunas para duas LEDs lado a lado).
8. Retire a peça boa e coloque a peça que será inspecionada, sem mover a câmera.
9. Clique em **Adicionar foto ao buffer** várias vezes, mudando levemente o ângulo ou aguardando um quadro mais nítido. O buffer aceita até 8 fotos.
10. Clique em **INSPECIONAR QUADRO ATUAL**. O sistema analisa todas as fotos e mantém o pior resultado de cada LED.
11. O resultado mostra os LEDs locais e o parecer da LLM como **OK** ou **REJEITADO**.

A inspeção e a chamada ao Ollama acontecem em segundo plano. A câmera continua mostrando imagens enquanto o modelo analisa a peça e fica pronta para a próxima inspeção.

## Ollama

O endereço padrão é `` e o modelo recomendado é `qwen3-vl:8b-instruct`.

Use **Conectar e buscar modelos** para carregar os modelos que possuem capacidade de visão. As imagens são enviadas somente ao servidor configurado.

Se a LLM estiver indisponível, a análise visual local continua funcionando e o resultado é identificado como `OK LOCAL` ou `REVISAR`.

## Interface Streamlit anterior

A versão web continua disponível em `app.py` e pode ser aberta separadamente:

```powershell
python -m streamlit run app.py
```

## Testes

```powershell
python -m unittest discover -s tests -v
```

## Limitações

Mantenha câmera, suporte, foco, distância e iluminação constantes. Inclinação, perspectiva, reflexos e mudanças fortes de luz podem gerar falsos alarmes. Para uso industrial, calibre o limite com peças boas e defeituosas conhecidas e confirme o resultado por inspeção elétrica ou humana.
