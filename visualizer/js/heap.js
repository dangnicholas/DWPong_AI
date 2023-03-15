
/*
Index-preserving heap utilities for partial 2D matrix sorting
Contributed by Max Milkert
*/

//number is the flattened index
var get_from_matrix = (matrix, number) => {
    let columns = matrix[0].length;
    let row = Math.floor(number / columns);
    let column = number % columns;
    return matrix[row][column];
}

// parent = i-1 // 2
//left = 2i +1
// right = 2i + 2
const max_heapify = (matrix, A, i) => {
    let l = 2 * i + 1;
    let r = 2 * i + 2;
    let largest = i;
    if (l < A.length && get_from_matrix(matrix, A[l]) > get_from_matrix(matrix, A[largest])) {
        largest = l;
    }
    if (r < A.length && get_from_matrix(matrix, A[r]) > get_from_matrix(matrix, A[largest])) {
        largest = r;
    }
    if (largest != i) {
        temp = A[i];
        A[i] = A[largest];
        A[largest] = temp;
        max_heapify(matrix, A, largest);
    }
}

var heap_extract_max = (matrix, A) => {
    const max = A[0];
    A[0] = A[A.length - 1];
    A.pop();
    max_heapify(matrix, A, 0);
    return max;
}

const build_max_heap = (matrix, A) => {
    let i = Math.floor(A.length / 2);
    while (i >= 0) {
        max_heapify(matrix, A, i);
        i -= 1;
    }
}

var arg_heapsort = (matrix, top_x) => {
    A = Array.from(Array(matrix.length * matrix[0].length).keys());
    build_max_heap(matrix, A);
    let args = [];
    for (let i = 0; i < top_x; i++) {
        args.push(heap_extract_max(matrix, A));
    }
    return args;
}

/*
Example consumer:
let matrix_to_sort = [[98, 97, 95, 76], [1, 2, 300.9, 4]]
args = arg_heapsort(matrix_to_sort, 3)
for (let i = 0; i < args.length; i++) {
    console.log(get_from_matrix(matrix_to_sort, args[i]))
}
*/