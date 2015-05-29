def read_data(filename):
    """Read the indicated file, which is a matrix file of the kind produced
    by matrix2dst and return two lists.  The first list consists of lists of
    word IDs, where each list corresponds to a particular gloss. The second
    list consists of distance matrices, giving distances between all words
    with a given gloss.  The two lists correspond in order (i.e. if they are
    combined with 'zip' the result is sensible)."""
    
    id_list = []
    matrix_list = []
    fp = open(filename)
    starting = False
    for line in fp:
        line = line.strip()
        if not line:
            continue
        if line == "#":
            # This indicates the end of a meaning class
            id_list.append(ids)
            # Replace diagonal zeros
            for i in range(0, len(matrix)):
                matrix[i][i] = 0.0
            matrix_list.append(matrix)
        elif len(line.split()) == 1:
            # This indicates the beginning of a new meaning class
            ids = []
            matrix = []
        else:
            # This is one row of a matrix
            ids.append(int(line.split()[0]))
            row =  [float(p) for p in line.split()[1:]]
            # Replace extrema with not-quite-so-extrema.
            # This avoids explosions with Beta likelihoods.
            # Zeros on diagonals are not problematic, and these
            # get replaced later.
            row = [r if r > 0.0 else 0.01 for r in row]
            row = [r if r < 1.0 else 0.99 for r in row]
            matrix.append(row)
    fp.close()
    return id_list, matrix_list

def get_cogids(id_list, partitions):
    """Given a list of lists of word IDs (like the first return value of
    read_data above) and a list of partitions (where each partition is a list
    of lists containing consecutive integers from 0 to n), where the two lists
    correspond in order, return a dictionary which maps from word IDs to
    arbitrary CogIDs in such a way that words which are grouped
    together in the partition have the same CogID."""
    results = {}
    cogid = 1
    for ids, part in zip(id_list, partitions):
        for chunk in part:
            for bit in chunk:
                results[ids[bit]] = cogid
            cogid += 1
    return results

def extend_csv(in_filename, out_filename, cogids, ref):

    fp_in = open(in_filename,"r")
    fp_out = open(out_filename,"w")

    first = True
    for line in fp_in:
        if line.startswith("#"):
            fp_out.write(line)
        else:
            if first:
                first = False
                line = line.strip() + "\t%s\n" % ref
            else:
                id_ = line.split()[0]
                line = line.strip() + "\t%s\n" % cogids[id_]
            fp_out.write(line)

    fp_in.close()
    fp_out.close()
