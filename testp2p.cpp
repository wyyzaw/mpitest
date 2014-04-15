#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define INF 1e8
#define index(i, j) (n * i) + j

int n, rank, noProcs, *dists, *local, *counts, *zeroes, *rowK;
double starttime,endtime;
void load();
void setupMaster();
void goMasterNode();
void goNode();
void doPrint();

int main( int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &noProcs);

	if (rank == 0){
		clock_t start,finish;
		double totaltime;
		start=clock();
		//starttime = MPI_Wtime();
		goMasterNode();
		doPrint();
		//endtime = MPI_Wtime();
		finish=clock();
		totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
		printf("That tooks %f secodes\n", totaltime);
		//printf("That tooks %f secodes\n", endtime-starttime);
	}
	else
		goNode();

	MPI_Finalize();
}

void load() {
	int i, j, tmp;
	char name[15];
	printf("input file: ");
	scanf("%s", name);
	freopen(name, "r", stdin);
	printf("\n");

	scanf("%d", &n);

	dists = (int *) malloc(sizeof(int) * n * n);

	for (i = 0; i < n; ++i) {
		for (j = 0; j < n; ++j){
			scanf("%d", &tmp);
			if (tmp == -1) tmp = INF;
			dists[index(i, j)] = tmp;
		}
	}

}

void setupMaster() {
	int rowsPerProc, i, start;
	load();

	rowsPerProc = (n / noProcs);

	counts = (int*) malloc(sizeof(int) * noProcs);
	zeroes = (int*) malloc(sizeof(int) * noProcs);
	local  = (int*) malloc(sizeof(int) * rowsPerProc * n);
	start = 0;
	for (i = 0; i <noProcs; ++i) {
		counts[i] = rowsPerProc; 
		counts[i] *= n;
		zeroes[i] = start;
		start = ((i + 1) * rowsPerProc * n);

	}
	if (n % noProcs != 0) {
		counts[noProcs - 1] += (n % noProcs) * n;
	}

}


void goMasterNode() {
	int i, j, k, lb, ub, owner, rowsPerProc, me, *rowKSpace;
	setupMaster();

	rowsPerProc = (n / noProcs);
	me = rowsPerProc;
	if ((n % noProcs != 0) && (me == noProcs - 1)) {
		me += (n % noProcs);
	}
	lb = (rowsPerProc) * rank;
	ub = (rowsPerProc) * (rank + 1) - 1;
	local  = (int*) malloc(sizeof(int) * me * n);
	rowKSpace = (int *) malloc(sizeof(int) * n);


	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Scatterv(dists, counts, zeroes, 
		         MPI_INT, local, counts[0], 
		         MPI_INT, 0, MPI_COMM_WORLD);	

	for (k = 0; k < n; ++k) {
		owner = k / (rowsPerProc);
		if (owner >= noProcs) owner = noProcs - 1;

		if (owner == rank)
			rowK = &local[(k - lb) * n];
		else
			rowK = rowKSpace;

		MPI_Bcast(rowK, n, MPI_INT, owner, MPI_COMM_WORLD);

		for (i = 0; i < me; ++i) {
			for (j = 0; j < n; ++j) {
				if (local[index(i, k)] + rowK[j] < local[index(i, j)])
					local[index(i, j)] = local[index(i, k)] + rowK[j];
			}
		}

	}
	free(rowKSpace);

    MPI_Gatherv(local, counts[0], MPI_INT, 
		        dists, counts, zeroes, 
		        MPI_INT, 0, MPI_COMM_WORLD);

    free(local);


}

void goNode() {
	int rowsPerProc, i, j, k, u, v, start, me, lb, ub, owner, *rowKSpace;

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	rowsPerProc = (n / noProcs);
	me = rowsPerProc;
	if ((n % noProcs != 0) && (rank == noProcs - 1)) {
		me += (n % noProcs);
	}

	local  = (int*) malloc(sizeof(int) * me * n);

	MPI_Scatterv(NULL, NULL, NULL, 
		         MPI_INT, local, me * n, 
		         MPI_INT, 0, MPI_COMM_WORLD);

	lb = (n / noProcs) * rank;
	ub = (n / noProcs) * (rank + 1) - 1;


	rowKSpace = (int *) malloc(sizeof(int) * n);

	for (k = 0; k < n; ++k) {
		owner = k / (rowsPerProc);
		if (owner >= noProcs) owner = noProcs - 1;

		if (owner == rank)
			rowK = &local[(k - lb) * n];
		else
			rowK = rowKSpace;

		MPI_Bcast(rowK, n, MPI_INT, owner, MPI_COMM_WORLD);

		for (i = 0; i < me; ++i) {
			for (j = 0; j < n; ++j) {
				if (local[index(i, k)] + rowK[j] < local[index(i, j)])
					local[index(i, j)] = local[index(i, k)] + rowK[j];
			}
		}
	}
	free(rowKSpace);

	MPI_Gatherv(local, me * n, MPI_INT, 
		        NULL, NULL, NULL, 
		        MPI_INT, 0, MPI_COMM_WORLD);

	free(local);

}

void doPrint() {
	int i, j;
	freopen("c:\\out.txt","w",stdout); 
	for (i = 0; i < n; ++i) {
		for (j = 0; j < n; ++j) {
			printf("%d ", dists[index(i, j)]);
		}
		printf("\n");
	}
	 freopen("CON", "w", stdout);
	
}