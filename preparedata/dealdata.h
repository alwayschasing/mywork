/*************************************************************************
	> File Name: dealdata.h
	> Author: ruihong
	> Mail: 
	> Created Time: 2016年12月20日 星期二 20时12分40秒
 ************************************************************************/

#ifndef _DEALDATA_H
#define _DEALDATA_H
#include<string>
#include<vector>
using namespace std;

class UBasedData; //user-based data
class IBasedData; //item-based data

class PrepareData{
public:
    PrepareData(string rawdatapath,int userhashsize,int itemhashsize);
    UBasedData* getUserBasedData();
    IBasedData* getItemBasedData();
//protected:
private:
    string datapath;
    int user_hash_size;
    int item_hash_size;
    char delim = ','; 
};
//the user-based data unit,each include the movie id ,rating,and the time
struct uunit{
    int movid;
    int rating;
    int time;
    uunit(int,int,int);
    bool operator<(const uunit &a){
        return time<a.time; 
    };
};
struct iunit{
    int userid;
    int rating;
    int time;
    iunit(int,int,int); 
};
class UBasedData {
public:
    //parmeter inclue hashsize which is the max userid
    UBasedData(int hs);
    int getUserNum();
    void setUserNum(int unum);

    vector<vector<uunit>*> uhashtable;
    ~UBasedData();
private:
    int usernumber = 0;
    int hashsize;
};
class IBasedData {
public:
    //parmeter inclue hashsize which is the max userid
    IBasedData(int hs);
    int getItemNum();
    void setItemNum(int unum);

    vector<vector<iunit>*> ihashtable;
    ~IBasedData();
private:
    int usernumber = 0;
    int hashsize; 
};

#endif
