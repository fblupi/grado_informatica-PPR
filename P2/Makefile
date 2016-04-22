all : bbpar bbseq

bbseq: bbseq.o libbb.o
	mpicxx -O3 bbseq.o libbb.o -o bbseq

bbseq.o: bbseq.cc
	mpicxx -O3 -c bbseq.cc

bbpar: bbpar.o libbb.o
	mpicxx -O3 bbpar.o libbb.o -o bbpar

bbpar.o: bbpar.cc
	mpicxx -O3 -c bbpar.cc

libbb.o: libbb.cc libbb.h
	mpicxx -O3 -c -Wno-unused-result libbb.cc

clean:
	/bin/rm -f *.o bbseq bbpar
