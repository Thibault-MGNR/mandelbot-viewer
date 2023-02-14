LIB = lib
BIN = bin

LIBS = $(wildcard $(LIB)/*.lib)
EXEC = fractal.exe
OUT = compile.bat

all:
	@echo nvcc .\src\main.cu $(LIBS) -I .\include\ -L .\lib\ -o $(BIN)\$(EXEC) > $(OUT)
	.\$(OUT)
	del "$(OUT)" "$(BIN)\fractal.lib" "$(BIN)\fractal.exp"