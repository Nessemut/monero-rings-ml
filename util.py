ordered_probability_positions = [11, 10, 9, 8, 1, 7, 6, 5, 4, 3, 2]

def predict(ring):
    for position in ordered_probability_positions:
        if ring[position-1] == 1:
            return position

def get_empirical_accuracy(x, y):
    test_length = len(x)
    correctly_predicted_count = 0
    for row in range(0, test_length):
        ring = x[row][2:23:2]
        if y[row] == predict(ring):
            correctly_predicted_count += 1

    return correctly_predicted_count/test_length

def get_predicted_noneleven_rings(x_test, y_test, y_pred):
    test_length = len(x_test)
    predicted_noneleven_rings = []
    for row in range(0, test_length):
        pred_tuple = y_test[row], y_pred[row]
        if pred_tuple != (11, 11) and pred_tuple[0] == pred_tuple[1]:
            outs = x_test[row][2:23:2]
            predicted_noneleven_rings.append([outs, y_test[row], y_pred[row]])
    return predicted_noneleven_rings
