#定义编译器变量
CC = g++

objects = run.o itemBased.o utility.o

CFLAGS = --std=c++11

KNNRecommend:$(objects)
	$(CC) $(CFLAGS) -o KNNRecommend $(objects) -lpthread

run.o:run.cpp knn.h utility.h
	$(CC) $(CFLAGS) -c run.cpp 

itemBased.o:itemBased.cpp knn.h utility.h
	$(CC) $(CFLAGS) -c itemBased.cpp -lpthread

utility.o:utility.cpp utility.h
	$(CC) $(CFLAGS) -c utility.cpp

clean:
	rm KNNRecommend $(objects)


