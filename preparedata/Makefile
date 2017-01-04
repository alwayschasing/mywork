#get the data we need
CC = g++
objects = main.o dealdata.o
CFLAGS = --std=c++11

getUBasedData:$(objects)
	$(CC) -o getUBasedData $(objects)

$(objects):%.o:%.cpp *.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm getUBasedData $(objects)
