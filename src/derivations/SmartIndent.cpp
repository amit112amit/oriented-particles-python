#include <iostream>     // std::cin, std::cout
#include <fstream>      // std::ifstream
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main (int argc, char* argv[]) {
    if ( argc < 2 ) { 
        std::cout<<"usage: "<< argv[0] << " <inputFile> <outputFile> <lineLimit> <lineContinuationStr>" << std::endl;
        return 0;
    }
    else {
        std::ifstream inputFile( argv[1] );
        std::ofstream outputFile( argv[2] );
        int lineLimit = atoi( argv[3] );
        char * lineBreak = argv[4];
        int lineBreakLen = strlen(lineBreak);
        std::vector<char> line;
        line.resize(lineLimit,'$');
        int lastOperatorIndex = -1;
        
        if ( !inputFile.is_open() )
            std::cout << "Could not open file." << std::endl;
        else {
            char x;
            int chrIdx = 0;
            while ( inputFile.get ( x ) ){
                if( chrIdx < lineLimit ){
                    if(x == '+' || x == '-'){
                        lastOperatorIndex = chrIdx;
                    }
                    line[chrIdx++] = x;
                }
                else{
                    for(int j=0; j <= lastOperatorIndex; j++){
                        outputFile << line[j];
                    }
                    outputFile << lineBreak << std::endl <<"        ";
                    std::vector<char> temp;
                    for(int j = lastOperatorIndex + 1; j < lineLimit; j++ ){
                        temp.push_back( line[j] );
                    }
                    line.clear();
                    line.resize( lineLimit, '$' );
                    chrIdx = 0;
                    for(int j=0; j < temp.size(); j++){
                        line[j] = temp[j];
                        chrIdx++;
                    }
                    line[chrIdx++] = x;
                }
            }            
        }
        for(int j=0; j < lineLimit; j++){
            if( line[j] != '$' )
                outputFile << line[j];
            else
                break;
        }
        inputFile.close();
        outputFile.close();
        return 0;
    }
}
