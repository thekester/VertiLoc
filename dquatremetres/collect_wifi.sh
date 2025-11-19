#!/bin/sh

X="$1"
Y="$2"

# Interface AP et MAC du client ESP32
IFACE="phy1-ap0"
ESP32_MAC="88:57:21:23:2F:30"

# Fichier de sortie
OUTFILE="${X}_${Y}.csv"

# CSV header
echo "Signal,Noise,signal_A1,signal_A2,signal_A3" > "$OUTFILE"

# Nombre de mesures et intervalle
INTERVAL=0.2
DURATION=5
STEPS=$(awk "BEGIN {print int($DURATION/$INTERVAL)}")

# Fonctions d'aide
get_signal_line() {
  # fetch the "signal: -XX [-a1, -a2, -a3] dBm" line if available
  iw dev "$IFACE" station get "$ESP32_MAC" 2>/dev/null | grep -m1 'signal:' \
  || iw dev "$IFACE" station dump 2>/dev/null | grep -A5 -i "$ESP32_MAC" | grep -m1 'signal:' \
  || iw dev "$IFACE" link 2>/dev/null | grep -m1 'signal:'
}

parse_signal_value() {
  # extract the dBm value after "signal:" (unit removed)
  awk '{for(i=1;i<=NF;i++) if($i=="signal:"||$i=="signal") {gsub(/[^0-9\.-]/,"",$(i+1)); print $(i+1); exit}}'
}

parse_per_chain() {
  # capture the content between brackets and split it into 3 values
  BRACKETS=$(sed -n 's/.*\[\(.*\)\].*/\1/p')
  if [ -n "$BRACKETS" ]; then
    A1=$(printf "%s" "$BRACKETS" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')
    A2=$(printf "%s" "$BRACKETS" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}')
    A3=$(printf "%s" "$BRACKETS" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$3); print $3}')
    echo "${A1:-NA},${A2:-NA},${A3:-NA}"
  else
    # no per-chain info, reuse the global signal for each antenna
    echo "$SIG,$SIG,$SIG"
  fi
}

get_noise() {
  # prefer `iw survey dump`, fallback to `iwinfo`
  N=$(iw dev "$IFACE" survey dump 2>/dev/null | awk '/noise:/ {v=$2} END{print v}')
  if [ -z "$N" ]; then
    command -v iwinfo >/dev/null 2>&1 && N=$(iwinfo "$IFACE" info 2>/dev/null | awk '/Noise:/ {print $2}')
  fi
  echo "${N:-NA}"
}

i=0
while [ $i -lt "$STEPS" ]; do
  # clean decimal timestamp
  TIME=$(awk -v i="$i" -v dt="$INTERVAL" 'BEGIN{printf("%.3f", i*dt)}')

  # read the signal line once, then parse
  LINE=$(get_signal_line)
  SIG=$(printf "%s\n" "$LINE" | parse_signal_value)

  # per-antenna RSSI
  CHAINS=$(printf "%s\n" "$LINE" | parse_per_chain)

  # noise
  NOISE=$(get_noise)

  # fallback if the signal is empty
  [ -z "$SIG" ] && SIG="NA"
  # ensure we always have 3 antenna values
  C1=$(echo "$CHAINS" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')
  C2=$(echo "$CHAINS" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}')
  C3=$(echo "$CHAINS" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$3); print $3}')
  [ -z "$C1" ] && C1="$SIG"
  [ -z "$C2" ] && C2="$SIG"
  [ -z "$C3" ] && C3="$SIG"

  echo "$SIG,$NOISE,$C1,$C2,$C3" >> "$OUTFILE"

  sleep "$INTERVAL"
  i=$((i+1))
done
