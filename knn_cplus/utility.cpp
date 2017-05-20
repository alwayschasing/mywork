/*************************************************************************
	> File Name: utility.cpp
	> Author: ruihong
	> Mail: 
	> Created Time: 2017年05月14日 星期日 16时34分28秒
 ************************************************************************/

#include"utility.h"
#include<vector>
#include<algorithm>
#include<string>
using namespace std;

vector<int> split2int(string line,char dim){
    int p = 0, len = line.size();
    vector<int> res;
    int start = 0;
    while(p<len){
        if(line[p] == dim){
            res.push_back(stoi(line.substr(start,p-start)));
            start = p+1;
        }
        p++;
    }
    return res;
}

//对数组的下标按数组的值进行排序
vector<int> argsort(vector<float> &a){
    int ln = a.size();
    vector<int> arg(a.size(),0);
    for(int i = 0; i < ln; ++i){
        arg[i] = i;
    }
    sort(arg.begin(),arg.end(),[&a](int p,int q){return a[p]>a[q];});
    return arg;
}



