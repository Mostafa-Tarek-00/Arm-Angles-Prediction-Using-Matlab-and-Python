
x = readtable('train.xlsx');

out = table2array(x(1:6,:));
inp = table2array(x(7:9,:)); 

load('net_new_ff.mat', 'net1');
outputs1 = sim(net1, inp);
error1 = out - outputs1;

load('net_new_grnn.mat','net2');

outputs2 = sim(net2, inp);
error2 = out - outputs2;

load('net_new_rb.mat', 'net3');

outputs3 = sim(net3, inp);
error3 = out - outputs3;

for i = 1:size(out, 1)
    figure(i)
    cc = 1:size(out, 2);
    plot(cc, out(i,:), 'ko', cc, outputs1(i,:), 'r^', cc, outputs2(i,:), 'xb', cc, outputs3(i,:), '*g');
    xlabel('No.'), ylabel(['Output ', num2str(i)]);
    legend('COMSOL results', 'fitted NNFF (net2)','fitted NEWGRNN (net2)', 'fitted NEWRB (net3)');
    title(['Output ', num2str(i), ' from COMSOL and Neural Networks']);
    saveas(gcf, ['output_train_', num2str(i), '.jpg']);
end

