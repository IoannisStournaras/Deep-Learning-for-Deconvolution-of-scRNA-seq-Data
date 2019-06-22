%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ������� ������������� CD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% �������� ��� �����������
%
  clear all;             % �������� ��� ����� ��������
  close all;             % �������� ���� ��� �������� �����������
  clc;                   % ���������� ��� ��������� �������

  tic;     % Reset and start the computers stop watch.
           % Starts a stopwatch timer to measure performance. 
           % The function records the internal time at execution of the tic
           % command. Display the elapsed time with the toc function.
                    
% E������� 3 seconds ����������� ������� ��� plots, ���� �� ������� � ����- 
% �������� ������ ��� �� �������� ��� �������� �����������.

% H ����������� �� ��������� �������� 721 sec (Elapsed time=721.006093 sec)

% ���������� ������� ���� �������� - O������ ����������
  SymbolRate = 12e9;        % Signal Frequency-������ ��������� ������� 12Gbps
  sps = 16;                 % samples per symbol
  SampleRate = 16*12e9;     % sample rate = sps * symbol rate
  Tsamples = 1/SampleRate;  % �� �s ��� ��� ���������� ��� ������
  Tsymbol  = 1/SymbolRate;
  Nsymbols = 10*2048;       % Number Of Symbols (������������ 10*2048 �������) 
  TimeWindow = Nsymbols/SymbolRate;
  
 % �������� ��������� - ���������� NRZ train pulse (h), DataPattern=PRBS7
   h = commsrc.pattern('SamplingFrequency', SampleRate, ...
                         'SamplesPerSymbol' , sps, ...
                         'PulseType', 'NRZ', ...
                         'OutputLevels', [-1 1], ...
                         'RiseTime', 0, ...
                         'FallTime', 0, ...
                         'DataPattern', 'PRBS7', ...
                         'Jitter', commsrc.combinedjitter);
    t = 0:Tsamples:TimeWindow-Tsamples;
    figure(1);
    Signal=generate(h,Nsymbols);

  % �������� �������� 1(����� ��� plot)
    plot(t(40*16:70*16),Signal(40*16:70*16),'b','LineWidth',3);
    xlabel('������');
    ylabel('������');
    title('������ ��������� ������ ��������� ��� ����� ��� ������');
    pause(3);        % pauses execution for 3 seconds before continuing

  % �������� ���������� ��������� ��� ���� ��� ����� ���������� 
  % ��� �/� Fourier (���������������� ��� ��������� ��� ������ ��� �� �����
  % ��� ������ ��� ����� ��� ���������� �� �� ������� ��� �/� Fourier)
    N = length(Signal);
    f=-(SampleRate/2):(SampleRate)/(N-1):(SampleRate/2); %������ ����������
    S = fftshift(fft(Signal,N));  % ������� �������� ��� ������������� ��� 
                                  % ����� (��� ���������� ����������)
    figure(2);
    f=f';

  % �������� �������� 2 (����� ��� plot)
    plot(f,abs(S),'y','LineWidth',3);
    xlabel('���������');
    ylabel('������');
    title('������ ��������� ������ ��������� ��� ����� ��� ����������');
    pause(3);

  % ���������� ��������� ��������� (�������� CD)
    W = 2*pi*f;                     % ������� ���������
    Z = 51e3;                       % Fiber Length
    D = 17e-6;                      % Fiber Dispersion
    Reference_Frequency = 193.1e12; % �� Hz
    c = 299792458;                  % ������� ���������� �� 3*(10^8)m/s
    Lref = c/Reference_Frequency;   % ����� ������� �=1.5525e-6 m
    B2 = -(Lref.^2)*D/(2*pi*c);      % ���������� GVD (�� �2 ������)

    H = exp(1i*(B2*Z/(2))*(W.^2)); % ��������� ��������� H(z,�) ��� CD

    SS=H.*S;     % ��������������� ��� ���������� ��������� ��� ����������
                 % ��������� �� ��� ���������
    figure(3);

  % �������� �������� 3 (����� ��� plot)
    plot(f,abs(SS),'g','LineWidth',3);
    xlabel('���������');
    ylabel('������');
    title('��������� ������ ��� ����� ��� ���������� ���� �� ��������');

    pause (3);

    ss=ifft(ifftshift(SS));  % ��������� ������� ��� ����� ��� ������ ���� 
                             % ����������� �/� Fourier
    figure(4);

  % �������� �������� 4 (����� ��� plot)
    plot(t(40*16:70*16), real(ss(40*16:70*16)),'r','LineWidth',3);
    xlabel('������');    
    ylabel('������');
    title('��������� ������ ��� ����� ��� ������ ���� �� ��������');

  % ��������� ��������� ������� (�������� ��� ������� ��� ��� ������������
  % ��� ���������� ���������). E�������� 8-10 ����������� ��� ������.
    N_taps= 2*floor((D*(Lref.^2)*Z)/(2*c*(Tsamples^2)))+1; % ������� �����
    b = zeros(1,length(N_taps));      % �������������
  for k = 1:N_taps                  % loop ��� ��������� �����
    arg=(1i*c*Tsamples^2)/(D*Lref.^2*Z);
    b(k)=sqrt(arg).*exp(-arg*pi*(k-round(N_taps/2))^2);
    A = k-round(N_taps/2);
  end
  % �������� ��������� kaiser (������ ��������� �����)
    w=b.*kaiser(N_taps)';       
    data=filter(w,1,ss);              % chromatic dispersion equalization
    data = data./ max(data);
    data = data((N_taps/2):end);      % ����� �� ����� �/2 taps
    figure(5);
    
  % �������� ���� �������� 4 (����� ��� plot)
    plot(t(40*16:70*16), real(data(40*16:70*16)),'c','LineWidth',3)             
    xlabel('������');
    ylabel('������');
    title('��������� ������ ��� ����� ��� ������ ��� ������������ CD');

  % ��������� �������� ��� ������������-������������� �������
    eyediagram(Signal,sps);
    xlabel('������');
    ylabel('������');
    title('��������� �������� ������������� �������');
    pause(3);
     
  % ��������� �������� ��� ��������� �������
    eyediagram(real(ss),sps);
    xlabel('������');
    ylabel('������');
    title('��������� �������� ��������� �������');
    pause(3);
     
  % ��������� �������� �� FIR ������
    eyediagram(real(data(N_taps:4000)),sps);
    xlabel('������');
    ylabel('������');
    title('��������� �������� �� FIR ������');
    pause(3);

  % ��������� Scatterplots ��� Signal,ss ��� FIR.
  % Display the scatter plot of the constellation (64-QAM).
    a=scatterplot(Signal,64,0,'*-b'); hold on; % �ffset=0 ��� ���������� ��
                                               % ��������� ���� ��������  
    scatterplot(ss,64,0,'.b',a); hold on;
    pause (3);
    scatterplot(data,64,0,'.g',a);
    xlabel('���������� �����');
    ylabel('���������� �����');
    title('Constellation Diagram');
    legend('���� ���������','������ ����','������������ CD','Location','northwest');
    pause (721-toc);
    toc;                           % print the time since the tic command









                    