/*
 * bgrad conv
 */

for (int i = 0; i < M; ++i) {
    db[i] = 0;
    for (int j = 0; j < N*E*F; ++j) {
        db[i] += dy[j*M+i];
    }
}