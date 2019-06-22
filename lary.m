%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Κώδικας αντιστάθμισης CD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Διαγράφή του παρελθόντος
%
  clear all;             % διαγραφή του χώρου εργασίας
  close all;             % κλείσιμο όλων των γραφικών παραστάσεων
  clc;                   % εκκαθάριση του παραθύρου εντολών

  tic;     % Reset and start the computers stop watch.
           % Starts a stopwatch timer to measure performance. 
           % The function records the internal time at execution of the tic
           % command. Display the elapsed time with the toc function.
                    
% Eισαγωγή 3 seconds καθυστέρηση ανάμεσα στα plots, ώστε να υπάρξει ο απαι- 
% τούμενος χρόνος για τη σύγκριση των γραφικών παραστάσεων.

% H προσομοίωση θα διαρκέσει συνολικά 721 sec (Elapsed time=721.006093 sec)

% Δημιουργία σήματος προς μετάδοση - Oρισμός μεταβλητών
  SymbolRate = 12e9;        % Signal Frequency-ρυθμός μετάδοσης σήματος 12Gbps
  sps = 16;                 % samples per symbol
  SampleRate = 16*12e9;     % sample rate = sps * symbol rate
  Tsamples = 1/SampleRate;  % το Τs απο την παρουσίαση στο μάθημα
  Tsymbol  = 1/SymbolRate;
  Nsymbols = 10*2048;       % Number Of Symbols (Δημιουργούμε 10*2048 σύμβολα) 
  TimeWindow = Nsymbols/SymbolRate;
  
 % Παραγωγή δεδομένων - δημιουργία NRZ train pulse (h), DataPattern=PRBS7
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

  % Σχεδίαση γραφικής 1(χρήση της plot)
    plot(t(40*16:70*16),Signal(40*16:70*16),'b','LineWidth',3);
    xlabel('Χρόνος');
    ylabel('Πλάτος');
    title('Αρχική ακολουθία παλμών μετάδοσης στο πεδίο του χρόνου');
    pause(3);        % pauses execution for 3 seconds before continuing

  % Εφαρμογή χρωματικής διασποράς στο σήμα στο πεδίο συχνότητας 
  % και Μ/Σ Fourier (Μετασχηματίζουμε την ακολουθία των παλμών από το πεδίο
  % του χρόνου στο πεδίο της συχνότητας με τη βοήθεια του Μ/Σ Fourier)
    N = length(Signal);
    f=-(SampleRate/2):(SampleRate)/(N-1):(SampleRate/2); %πλέγμα συχνοτήτων
    S = fftshift(fft(Signal,N));  % κυκλική ολίσθηση και κεντραρισμένα στο 
                                  % μηδέν (για αμφίπλευρα διαστήματα)
    figure(2);
    f=f';

  % Σχεδίαση γραφικής 2 (χρήση της plot)
    plot(f,abs(S),'y','LineWidth',3);
    xlabel('Συχνότητα');
    ylabel('Πλάτος');
    title('Αρχική ακολουθία παλμών μετάδοσης στο πεδίο της συχνότητας');
    pause(3);

  % Δημιουργία Συνάρτηση Μεταφοράς (Εισαγωγή CD)
    W = 2*pi*f;                     % κυκλική συχνότητα
    Z = 51e3;                       % Fiber Length
    D = 17e-6;                      % Fiber Dispersion
    Reference_Frequency = 193.1e12; % σε Hz
    c = 299792458;                  % συνήθως λαμβάνεται ως 3*(10^8)m/s
    Lref = c/Reference_Frequency;   % μήκος κύματος λ=1.5525e-6 m
    B2 = -(Lref.^2)*D/(2*pi*c);      % παράμετρος GVD (το β2 δηλαδή)

    H = exp(1i*(B2*Z/(2))*(W.^2)); % συναρτηση μεταφοράς H(z,ω) της CD

    SS=H.*S;     % Πολλαπλασιασμός της συνάρτησης μεταφοράς της χρωματικής
                 % διασποράς με την ακολουθία
    figure(3);

  % Σχεδίαση γραφικής 3 (χρήση της plot)
    plot(f,abs(SS),'g','LineWidth',3);
    xlabel('Συχνότητα');
    ylabel('Πλάτος');
    title('Ακολουθία παλμών στο πεδίο της συχνότητας μετά τη μετάδοση');

    pause (3);

    ss=ifft(ifftshift(SS));  % μετατροπή σήματος στο πεδίο του χρόνου μεσω 
                             % αντίστροφου Μ/Σ Fourier
    figure(4);

  % Σχεδίαση γραφικής 4 (χρήση της plot)
    plot(t(40*16:70*16), real(ss(40*16:70*16)),'r','LineWidth',3);
    xlabel('Χρόνος');    
    ylabel('Πλάτος');
    title('Ακολουθία παλμών στο πεδίο του χρόνου μετά τη μετάδοση');

  % Κρουστική συνάρτηση φίλτρου (Εφαρμογή του φίλτρου για την αντιστάθμιση
  % της χρωματικής διασποράς). Eξισώσεις 8-10 παρουσίασης στο μάθημα.
    N_taps= 2*floor((D*(Lref.^2)*Z)/(2*c*(Tsamples^2)))+1; % αριθμός βαρών
    b = zeros(1,length(N_taps));      % κανονικοπίηση
  for k = 1:N_taps                  % loop για υλοποίηση βαρών
    arg=(1i*c*Tsamples^2)/(D*Lref.^2*Z);
    b(k)=sqrt(arg).*exp(-arg*pi*(k-round(N_taps/2))^2);
    A = k-round(N_taps/2);
  end
  % Εφαρμογή παραθύρου kaiser (μείωση πλευρικών λοβών)
    w=b.*kaiser(N_taps)';       
    data=filter(w,1,ss);              % chromatic dispersion equalization
    data = data./ max(data);
    data = data((N_taps/2):end);      % αγνοώ τα πρώτα Ν/2 taps
    figure(5);
    
  % Σχεδίαση νέας γραφικής 4 (χρήση της plot)
    plot(t(40*16:70*16), real(data(40*16:70*16)),'c','LineWidth',3)             
    xlabel('Χρόνος');
    ylabel('Πλάτος');
    title('Ακολουθία παλμών στο πεδίο του χρόνου για αντιστάθμιση CD');

  % Διαγράμμα οφθαλμού του εκπεμπόμενου-μεταδιδόμενου σήματος
    eyediagram(Signal,sps);
    xlabel('Χρόνος');
    ylabel('Πλάτος');
    title('Διαγράμμα οφθαλμού μεταδιδόμενου σήματος');
    pause(3);
     
  % Διαγράμμα οφθαλμού του ληφθέντος σήματος
    eyediagram(real(ss),sps);
    xlabel('Χρόνος');
    ylabel('Πλάτος');
    title('Διαγράμμα οφθαλμού ληφθέντος σήματος');
    pause(3);
     
  % Διαγράμμα οφθαλμού με FIR φίλτρο
    eyediagram(real(data(N_taps:4000)),sps);
    xlabel('Χρόνος');
    ylabel('Πλάτος');
    title('Διαγράμμα οφθαλμού με FIR φίλτρο');
    pause(3);

  % Υλοποίηση Scatterplots των Signal,ss και FIR.
  % Display the scatter plot of the constellation (64-QAM).
    a=scatterplot(Signal,64,0,'*-b'); hold on; % οffset=0 και απεικόνιση με
                                               % αστεράκια μπλε χρώματος  
    scatterplot(ss,64,0,'.b',a); hold on;
    pause (3);
    scatterplot(data,64,0,'.g',a);
    xlabel('πραγματικό μέρος');
    ylabel('φανταστικό μέρος');
    title('Constellation Diagram');
    legend('Σήμα μετάδοσης','Ληφθέν σήμα','Αντιστάθμιση CD','Location','northwest');
    pause (721-toc);
    toc;                           % print the time since the tic command









                    