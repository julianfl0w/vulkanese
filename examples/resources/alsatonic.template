/*
 * File: alsatonic.c
 * Tone generator from alsa audio api
 * Author: Coolbrother
 * Date: 24/12/2020
 * */

#include <alsa/asoundlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#define BUF_LEN 48000
#define DEF_FREQ 440
#define DEF_DUR 1
#define DEF_NOTE 48
#define MAX_NOTE 87
float g_buffer[BUF_LEN];
snd_pcm_t *g_handle;
snd_pcm_sframes_t g_frames;
int channels =1;
snd_pcm_format_t format = SND_PCM_FORMAT_FLOAT;
int rate = 48000;


#define HELP_TEXT "AlsaTonic Usage:\n\
  freq dur : without options, play at frequency freq, in duration dur in seconds.\n\
  -d dur : set duration in seconds (default: 1 sec)\n\
  -f freq : set frequency in HZ, and play it. (default: 440 HZ)\n\
  -F freq : set frequency in HZ, and play the sequence freq between start and stop optionss\n\
  -h : print this Help\n\
  -n note : set the note number and play it. (default: 48)\n\
  -N note : set note number and play the notes sequence between start and stop options\n\
  -s start : set start frequency or note for sequence (default: 0)\n\
  -S stop : set stop frequency or note for sequence (default: 1)\n\
  -t step : set step frequency or note for sequence (default: 1)\n\n"
//-----------------------------------------

float* genTone(float freq) {
    // must create a pointer to return buffer
    // float *buf = malloc(sizeof(float) * BUF_LEN);
    float vol =1;
    float t = 2*M_PI*freq/(rate*channels);
    // int nbSamples = rate * channels * dur;
    // printf("nbSamples: %d\n", nbSamples);
    int maxSamp = 32767;
    float* buf = g_buffer;
    for (int i=0; i< BUF_LEN; i++) {
        // val = (int)maxSamp*vol*sin(t*i);
        g_buffer[i] = sin(t*i);
    }

    return g_buffer;
}
//-----------------------------------------

int openDevice() {
    // open sound device and set params
 	  const static char *device = "default";
	  snd_output_t *output = NULL;
   int err;

    if ((err = snd_pcm_open(&g_handle, device, SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
      fprintf(stderr, "AlsaTonic: Playback open error: %s\n", snd_strerror(err));
      return EXIT_FAILURE;
    }

    if ((err = snd_pcm_set_params(g_handle,
        format,
        // SND_PCM_FORMAT_S16_LE,
        SND_PCM_ACCESS_RW_INTERLEAVED,
        channels,
        // BUF_LEN,
        rate,
        1, /* period */
        500000)) < 0) {	 /* latency: 0.5sec */ 
      fprintf(stderr, "AlsaTonic: Playback open error: %s\n", snd_strerror(err));
      return EXIT_FAILURE;
    }

}
//-----------------------------------------

void closeDevice() {
    // closing sound device
    // necessary to flush and send short sample
    snd_pcm_drain(g_handle);
	  snd_pcm_close(g_handle);

}
//-----------------------------------------

void writeBuf(float* buf, int nbFrames, int nbTimes) {
  for (int i=0; i < nbTimes; i++) {
      // Sending the sound
      g_frames += snd_pcm_writei(g_handle, buf, nbFrames);
  }
  // printf("WriteBuf nbFrames: %d\n", g_frames);

}
//-----------------------------------------

void writeAudio(unsigned int nbFrames) {
    /// Not used, just for notes
    // Sending the sound
    int frames = snd_pcm_writei(g_handle, g_buffer, nbFrames);
}
//-----------------------------------------

void playFreq(float freq, float dur) {
    // playing one freq
    float* buf;
    int nbSamples = rate * channels * dur;
    int nbTimes = nbSamples / BUF_LEN;
    int restFrames = nbSamples % BUF_LEN;
    // printf("restFrames: %d\n", restFrames);
    if (nbSamples >0) {
        buf = genTone(freq);
        if (nbTimes >0) {
            writeBuf(buf, BUF_LEN, nbTimes);	
        }
        if (restFrames > 0) {
            writeBuf(buf, restFrames, 1);
        }
    
    }
    // printf("nbFrames: %d\n", g_frames);

}
//-----------------------------------------

void playSeq(float freq, float dur, int start, int stop, float step) {
    // playing sequence freq
    // step is in float for frequency
    int iStep = (int)(step);
    for (int i=start; i<stop; i += iStep){
        playFreq(freq, dur);	
        freq += step;
    }

}
//-----------------------------------------

void playNote(int numNote, float dur) {
    // playing note number
    float refNote = 27.5; // note reference A0
    // test whether note between 0 and note_max
    numNote = (numNote < 0) ? DEF_NOTE : (numNote > MAX_NOTE) ? MAX_NOTE : numNote;
    float freq = refNote*pow(2, numNote*1/12.0);
    playFreq(freq, dur);	
    printf("Freq: %.3f\n", freq);


}
//-----------------------------------------

void playSeqNote(int numNote, float dur, int start, int stop, int step) {
    // playing sequence notes
    for (int i=start; i<stop; i += step) {
        playNote(numNote, dur);	
        numNote += step;
    }

}
//-----------------------------------------


int main(int argc, char *argv[]) {
    int err;
   
    float freq = DEF_FREQ; // in hertz
    float dur = DEF_DUR; // in seconds
    int note = DEF_NOTE;
    int start =0;
    int stop =1;
    float step =1;
    int mode =0;
    int optIndex =0;

    while (( optIndex = getopt(argc, argv, "d:f:F:hn:N:s:S:t:")) != -1) {
        switch (optIndex) {
            case 'd':
                dur = atof(optarg); break;
            case 'f':
                mode =1;
                freq = atof(optarg); break;
            case 'F':
                mode =2;
                freq = atof(optarg); break;
            case 'h':
                printf(HELP_TEXT);
                return 0;
            case 'n':
                mode =3;
                note = strtol(optarg, NULL, 10); break;
            case 'N':
                mode =4;
                note = strtol(optarg, NULL, 10); break;
            case 's':
                start = strtol(optarg, NULL, 10); break;
            case 'S':
                stop = strtol(optarg, NULL, 10); break;
            case 't':
                step = atof(optarg); break;

            default:
                printf("Option incorrect\n");
                return 1;
          }
      
    }
    
    // SINE WAVE
    if (err = openDevice()) {
        return EXIT_FAILURE;
    }
    
    // playing mode
    // without options
    if (mode == 0) {
        freq = (argc > 1) ? atof(argv[1]) : DEF_FREQ;
        if (freq == 0) {
            fprintf(stderr, "AlsaTonic: Invalid frequency.\n");
            return EXIT_FAILURE;
        }

        dur = (argc > 2) ? atof(argv[2]) : DEF_DUR;
        if (dur == 0) {
            fprintf(stderr, "AlsaTonic: Invalid duration.\n");
            return EXIT_FAILURE;
        }
        
        start = (argc > 3) ? strtol(argv[3], NULL, 10) : -1;
        stop = (argc > 4) ? strtol(argv[4], NULL, 10) : 0;
        step = (argc > 5) ? atof(argv[5]) : 1;

        // Playing freq
        if (start == -1) {
            printf("Playing freq, Sine tone at %.3fHz during %.3f secs.\n", freq, dur);
            playFreq(freq, dur);
        // Playing sequence freq
        } else {
            printf("Playing SeqFreq, Sine tone at %.3fHz, during %.3f secs, start: %d, stop: %d, step: %.3f.\n", freq, dur, start, stop, step);
            playSeq(freq, dur, start, stop, step);
        }

    } else if (mode == 1) {
        printf("Playing Freq, Sine tone at %.3fHz during %.3f sec.\n", freq, dur);
        playFreq(freq, dur);
    } else if (mode == 2) {
        printf("Playing SeqFreq, Sine tone at %.3fHz, during %.3f secs, start: %d, stop: %d, step: %.3f.\n", freq, dur, start, stop, step);
        playSeq(freq, dur, start, stop, step);

    } else if (mode == 3) {
        printf("Playing Note at %d, during %.3f secs.\n", note, dur);
        playNote(note, dur);
    } else if (mode == 4) {
        printf("Playing sequence Note at note: %d, during %.3f secs, start: %d, stop: %d, step: %.3f.\n", note, dur, start, stop, step);
        playSeqNote(note, dur, start, stop, step);
    } 
     
    printf("nbFrames played: %d\n", g_frames);


    closeDevice();

    return EXIT_SUCCESS;
}
//-----------------------------------------
