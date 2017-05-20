/*************************************************************************
	> File Name: test.cpp
	> Author: 
	> Mail: 
	> Created Time: Tue 16 May 2017 10:51:32 PM EDT
 ************************************************************************/

#include<iostream>
#include<fstream>
#include<string>
#include<vector>
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
void getMatrix(string path){
    ifstream input(path.c_str());
    int count = 0;
    string line = "";
    while(getline(input,line)){
        count++;
        //getline(input,line); 
        //if(line.size() == 0){
            //cout<<"zhaodaol"<<endl;
        //}
        //cout<<line<<endl;
        vector<int> temp = split2int(line,','); 
        int user = temp[0];
        int n = temp.size()/2;
        for(int i = 0; i < n; ++i){
            int item = temp[(i<<1)+1];
            int index = i<<2+2;
            int rating = temp[index];
            if(user == 889){
                cout<<i<<endl;
                cout<<temp.size()<<endl;
                cout<<(i*4)+2<<endl;
                cout<<"----"<<endl;
            }
            //ratingMat.matrix[user][item] = rating;
        }
    } 
}
int main(){
    string path = "/home/lrh/graduation_project/data/ml-1m/userbased.trainMF.csv";
    getMatrix(path);
    return 0;
}
