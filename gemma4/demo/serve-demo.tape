# VHS tape (https://github.com/charmbracelet/vhs) for serve-demo.gif: the Ollama CLI on the host asks gemma4_serve.py on the board.
# A template: make_demo.sh fills in @OUT@, @HOST@, @MODEL@ and @PROMPT@, starts the server and converts the recording.
# A dark ~87 x 20 terminal in a window, recorded at 960 x 540 (make_demo.sh scales it to 480 x 270), like bonsai2-27b/serve-demo.gif.
Output @OUT@
Set Shell zsh
Set FontFamily "Menlo"
Set FontSize 16
Set Width 960
Set Height 540
Set Padding 14
Set Margin 18
Set MarginFill "#000000"
Set WindowBar Colorful
Set WindowBarSize 30
Set BorderRadius 10
Set Theme { "name": "Terminal dark", "background": "#1e1e1e", "foreground": "#e8e8e8", "cursor": "#a0a0a0", "selection": "#3a3a3a", "black": "#000000", "red": "#c23621", "green": "#25bc24", "yellow": "#adad27", "blue": "#492ee1", "magenta": "#d338d3", "cyan": "#33bbc8", "white": "#cbcccd", "brightBlack": "#818383", "brightRed": "#fc391f", "brightGreen": "#31e722", "brightYellow": "#eaec23", "brightBlue": "#5833ff", "brightMagenta": "#f935f8", "brightCyan": "#14f0f0", "brightWhite": "#e9ebeb" }
Set TypingSpeed 35ms
Set Framerate 25
Set WaitTimeout 300s

Hide
Type "cd ~ && export PROMPT='%n@%m %1~ %# ' && clear"
Enter
Sleep 300ms
Show

Sleep 800ms
Type `OLLAMA_HOST=http://@HOST@:8000 ollama run --verbose @MODEL@ "@PROMPT@"`
Sleep 400ms
Enter
# the answer streams at the board's speed; the client prints its timings and exits: the prompt is back on the last line
Wait+Line /~ %\s*$/
Sleep 5s
