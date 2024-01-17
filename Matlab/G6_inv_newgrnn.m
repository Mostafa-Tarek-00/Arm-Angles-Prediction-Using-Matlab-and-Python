%clear all
x = xlsread('train.xlsx');
y = xlsread('test.xlsx');
format longe

inptrain = [x(7, :); x(8, :); x(9, :)];
inptest = [y(7, :); y(8, :); y(9, :)];

outtrain = [x(1, :); x(2, :); x(3, :); x(4, :); x(5, :); x(6, :)];
outtest = [y(1, :); y(2, :); y(3, :); y(4, :); y(5, :); y(6, :)];

%net2 = newgrnn(inptrain, outtrain, 0.001);
%save('net_new_grnn.mat', 'net2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
load('net_new_grnn.mat');

p1 = sim(net2, inptrain);
error1 = p1 - outtrain;

input_size = net2.inputs{1}.size;
output_size = net2.outputs{end}.size;

fprintf('Input Size of the Neural Network: %d\n', input_size);
fprintf('Output Size of the Neural Network: %d\n', output_size);

predictions = sim(net2, inptest);

mseValue = mse(outtest, predictions);

maeValue = mae(outtest, predictions);

rmseValue = sqrt(mseValue);

fprintf('Mean Squared Error (MSE): %.4f\n', mseValue);
fprintf('Mean Absolute Error (MAE): %.4f\n', maeValue);
fprintf('Root Mean Squared Error (RMSE): %.4f\n', rmseValue);

fprintf('\nTrue and Predicted Values for the First 5 Data Points:\n');
for i = 1:5
    fprintf('Data Point %d\n', i);
    fprintf('Input Values: %s\n', mat2str(inptest(:, i)));
    fprintf('True Values: %s\n', mat2str(outtest(:, i)));
    fprintf('Predicted Values: %s\n', mat2str(predictions(:, i)));
    fprintf('------------------------\n');
end
