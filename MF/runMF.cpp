/*************************************************************************
	> File Name: runMF.cpp
	> Author: 
	> Mail: 
	> Created Time: Sat Mar  4 06:56:43 2017
 ************************************************************************/

#include<iostream>
#include<fstream>
#include"./libmf-2.01/mf.h"

using namespace std;

int main(){
    mf::mf_node node = {1,2,4}; 
    ifstream input("~/graduation_project/data/ml-1m/mf/trainForMFlib"); 
    //读入训练数据
    mf::mf_problem model = {}
    while(!input.eof()){
        input>>
    }
    return 0;
}
