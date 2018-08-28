INCLUDE_DIR = -I/home/zqp/install_lib/include
INCLUDE_DIR += -I./include
INCLUDE_DIR += -I/usr/local/cuda/include

LIBRARY_DIR= -L/home/zqp/install_lib/lib

SRC = ${shell ls ./testDllPlateDtection2015/*.cpp}
SRC += ${shell ls ./src/*.cpp}

main: ${SRC}
	g++ -std=c++11 -o main ${SRC} ${INCLUDE_DIR} -lcaffe ${LIBRARY_DIR} `pkg-config --libs opencv` -lglog -lboost_system -lplateCharactRecognize

clean:
	-rm -rf main lib
