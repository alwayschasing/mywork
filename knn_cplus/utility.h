/*************************************************************************
	> File Name: utility.h
	> Author: ruihong
	> Mail: 
	> Created Time: 2017年05月14日 星期日 16时29分24秒
 ************************************************************************/

#ifndef _UTILITY_H
#define _UTILITY_H
#include<string>
#include<vector>
using namespace std;
//将一行字符串数据分裂为int类型的数组返回
vector<int> split2int(string line,char dim);

vector<int> argsort(vector<float> &a);
#endif
