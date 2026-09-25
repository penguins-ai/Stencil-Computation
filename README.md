## Build and Run

### Sequential Program

Compile the program:

```bash
gcc -O2 -o heat2d_serial heat2d_serial.c -lm
```

Run the program:

```bash
./heat2d_serial <N> <timesteps>
```

Example:

```bash
./heat2d_serial 1000 500
```


### Parallel Program

Compile the program:

```bash
gcc -O2 -fopenmp -o heat2d_omp heat2d_omp.c -lm
```

Run the program:

```bash
./heat2d_omp <N> <timesteps> [num_threads]
```
