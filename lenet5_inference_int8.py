import numpy as np

def relu_int8(x):
    return np.maximum(x, 0).astype(np.int8)

def softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def prediction(image,
              w_conv1, w_conv2, w_fc1, w_fc2, w_fc3,
              b_conv1, b_conv2, b_fc1, b_fc2, b_fc3):
    
    # Convert input image to int8
    image_int8 = (image * 127.0).astype(np.int8)
    
    # Conv1 layer
    conv1_out = np.zeros((6, 28, 28), dtype=np.int8)
    for c in range(6):
        temp = image_int8 * w_conv1[c][0][0] + b_conv1[c]
        conv1_out[c] = np.clip(temp, -128, 127).astype(np.int8)
    
    # ReLU
    conv1_out = relu_int8(conv1_out)
    
    # Pool1
    pool1_out = np.zeros((6, 14, 14), dtype=np.int8)
    for c in range(6):
        for h in range(14):
            for w in range(14):
                sum_val = (int(conv1_out[c, 2*h, 2*w]) + 
                          int(conv1_out[c, 2*h+1, 2*w]) +
                          int(conv1_out[c, 2*h, 2*w+1]) +
                          int(conv1_out[c, 2*h+1, 2*w+1]))
                pool1_out[c,h,w] = sum_val // 4
    
    # Conv2 layer
    conv2_out = np.zeros((16, 10, 10), dtype=np.int8)  # Thay đổi kích thước output
    for c in range(16):
        for h in range(10):  # Thay đổi range
            for w in range(10):  # Thay đổi range
                sum_val = 0
                for k in range(6):
                    for m in range(5):
                        for n in range(5):
                            if h+m < 14 and w+n < 14:  # Thêm điều kiện check bounds
                                sum_val += int(pool1_out[k,h+m,w+n]) * int(w_conv2[c,k,m,n])
                sum_val += b_conv2[c]
                conv2_out[c,h,w] = np.clip(sum_val, -128, 127)
    
    # ReLU
    conv2_out = relu_int8(conv2_out)
    
    # Pool2
    pool2_out = np.zeros((16, 5, 5), dtype=np.int8)
    for c in range(16):
        for h in range(5):
            for w in range(5):
                sum_val = (int(conv2_out[c,2*h,2*w]) +
                          int(conv2_out[c,2*h+1,2*w]) +
                          int(conv2_out[c,2*h,2*w+1]) +
                          int(conv2_out[c,2*h+1,2*w+1]))
                pool2_out[c,h,w] = sum_val // 4
    
    # Flatten
    flat_out = pool2_out.reshape(-1)
    
    # FC1
    fc1_out = np.zeros(120, dtype=np.int8)
    for i in range(120):
        sum_val = 0
        for j in range(400):
            sum_val += int(flat_out[j]) * int(w_fc1[i,j])
        sum_val += b_fc1[i]
        fc1_out[i] = np.clip(sum_val, -128, 127)
    fc1_out = relu_int8(fc1_out)
    
    # FC2
    fc2_out = np.zeros(84, dtype=np.int8)
    for i in range(84):
        sum_val = 0
        for j in range(120):
            sum_val += int(fc1_out[j]) * int(w_fc2[i,j])
        sum_val += b_fc2[i]
        fc2_out[i] = np.clip(sum_val, -128, 127)
    fc2_out = relu_int8(fc2_out)
    
    # FC3
    fc3_out = np.zeros(10, dtype=np.float32)
    for i in range(10):
        sum_val = 0
        for j in range(84):
            sum_val += int(fc2_out[j]) * int(w_fc3[i,j])
        sum_val += b_fc3[i]
        fc3_out[i] = float(sum_val) / 127.0
    
    # Softmax
    probs = softmax(fc3_out)
    return probs

def print_stats(layer_name, data):
    min_val = np.min(data)
    max_val = np.max(data)
    avg_val = np.mean(data)
    print(f"{layer_name} stats - Min: {min_val}, Max: {max_val}, Avg: {avg_val:.2f}")

# Main execution
if __name__ == "__main__":
    # Load weights and biases from files
    weights_dir = "data/weights_int8/"
    
    w_conv1 = np.loadtxt(f"{weights_dir}w_conv1.txt", dtype=np.int8).reshape(6,1,1)
    w_conv2 = np.loadtxt(f"{weights_dir}w_conv2.txt", dtype=np.int8).reshape(16,6,5,5)
    w_fc1 = np.loadtxt(f"{weights_dir}w_fc1.txt", dtype=np.int8).reshape(120,400)
    w_fc2 = np.loadtxt(f"{weights_dir}w_fc2.txt", dtype=np.int8).reshape(84,120)
    w_fc3 = np.loadtxt(f"{weights_dir}w_fc3.txt", dtype=np.int8).reshape(10,84)
    
    b_conv1 = np.loadtxt(f"{weights_dir}b_conv1.txt", dtype=np.int8)
    b_conv2 = np.loadtxt(f"{weights_dir}b_conv2.txt", dtype=np.int8)
    b_fc1 = np.loadtxt(f"{weights_dir}b_fc1.txt", dtype=np.int8)
    b_fc2 = np.loadtxt(f"{weights_dir}b_fc2.txt", dtype=np.int8)
    b_fc3 = np.loadtxt(f"{weights_dir}b_fc3.txt", dtype=np.int8)
    
    # Load test dataset
    test_images = np.loadtxt("mnist-test-image.txt", dtype=np.float32).reshape(-1, 28, 28)
    test_labels = np.loadtxt("mnist-test-target.txt", dtype=np.int32)
    
    # Inference
    correct = 0
    for i in range(len(test_images)):
        probs = prediction(test_images[i],
                         w_conv1, w_conv2, w_fc1, w_fc2, w_fc3,
                         b_conv1, b_conv2, b_fc1, b_fc2, b_fc3)
        pred = np.argmax(probs)
        if pred == test_labels[i]:
            correct += 1
        print(f"Predicted label: {pred}")
        print(f"Prediction: {correct}/{i+1}")
    
    print(f"Accuracy = {correct/len(test_images):.4f}")
    
    # Print weight statistics
    print("=== Weight Statistics ===")
    print_stats("Conv1", w_conv1)
    print_stats("Conv2", w_conv2)
    print_stats("FC1", w_fc1)
    print_stats("FC2", w_fc2)
    print_stats("FC3", w_fc3)
