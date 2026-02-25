#include <iostream>
#include <string>
#include <algorithm>

using namespace std;


bool isSubsequence(const string& S, const string& T) {
    int i = 0, j = 0;
    while (i < S.length() && j < T.length()) {
        if (S[i] == T[j]) {
            j++;
        }
        i++;
    }
    return j == T.length();
}

int main() {
    string S, T;
    cin >> S >> T;

    string current = S;
    int day = 1;

    while (true) {
        if (isSubsequence(current, T)) {
            cout << day << endl;
            break;
        }

        // Reverse current string and append original S
        reverse(current.begin(), current.end());
        current += S;
        day++;
    }

    return 0;
}
